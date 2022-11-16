from scipy.optimize import minimize, LinearConstraint
from scipy.special import softmax
import torch as th
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax as torch_softmax
import collections
import gpytorch as gp

th.set_default_dtype(th.float32)
    
class RegularizedWeaver(th.nn.Module):
    def __init__(self):
        super().__init__()
        
    def minimize_lagrangian_wrt_p(self, a, b, Delta, p_init, p_tilde, rho, y, max_iters=100, eps=1e-4, verbose=False, report=1):
        # TODO: stable weaver algorithm implemented in torch
        zero_tensor = th.tensor([0.])

        y_plus = th.maximum(y, zero_tensor)
        y_minus = th.minimum(y, zero_tensor)

        p = p_init
        s = (th.sum(a) + th.sum(b)) * 2

        for it in range(max_iters):
            # d_prime_v = rho * d_prime(p, p_tilde)
            # Trick to move the negative vs positive terms into the right side of the update equation
            # d_prime_plus = th.maximum(d_prime_v, zero_tensor)
            # d_prime_minus = th.minimum(d_prime_v, zero_tensor)

            tau = (th.matmul(p.view(1,-1), Delta)).t().flatten()
            mask = (tau != 0)
            out = th.zeros_like(tau)
            out[mask] = b[mask] / tau[mask]
                        
            # tau = b / np.transpose(p.T @ Delta)   # Are we doing computations on the items not in the connected components here?
            tau_plus = th.maximum(out, zero_tensor)
            tau_minus = th.minimum(out, zero_tensor)
            
            p_minus_ptilde = p - p_tilde # Should we only compute this for the entries with non-zero
            sub_grad_p_minus_ptilde = th.sign(p_minus_ptilde)
            sub_grad_p_minus_ptilde_plus = th.maximum(sub_grad_p_minus_ptilde, zero_tensor)
            sub_grad_p_minus_ptilde_minus = th.minimum(sub_grad_p_minus_ptilde, zero_tensor)
            p_minus_ptilde_plus = th.maximum(p_minus_ptilde, zero_tensor)
            p_minus_ptilde_minus = th.minimum(p_minus_ptilde, zero_tensor)
            
            # denom = (s - th.matmul(Delta, tau_minus).flatten() - y_minus - d_prime_minus)
            # denom = s - th.matmul(Delta, tau_minus).flatten()
            denom = s * th.ones_like(p) - th.matmul(Delta, tau_minus).flatten() + y_plus
            mask = (denom != 0)
            denom[mask] = denom[mask] + rho * sub_grad_p_minus_ptilde_plus[mask]
            # denom[mask] = denom[mask] + rho * p_minus_ptilde_plus[mask]
            
            p_next = th.zeros_like(p)
            # numer = a + (th.matmul(Delta, tau_plus).flatten() - y_minus) * p + rho * p_tilde # This KL divergence regularization should still preserve the ascent property?
            numer = a + (th.matmul(Delta, tau_plus).flatten() - y_minus) * p
            
            numer[mask] = numer[mask] - rho * sub_grad_p_minus_ptilde_minus[mask] * p[mask]
            # numer[mask] = numer[mask] - rho * p_minus_ptilde_minus[mask] * p[mask]
            
            p_next[mask] = numer[mask]/denom[mask]
            
            p_next = p_next / th.sum(p_next)
            # print(th.sum(p_next))
            if not th.abs(th.sum(p_next) - 1) < 1e-6:
                print(th.sum(p_next))
            assert(th.abs(th.sum(p_next) - 1) < 1e-6)
            
            if verbose and it % report == 0:
                mask_a = (p!= 0)
                log_p = th.zeros_like(p)
                log_p[mask_a] = th.log(p[mask_a])
                Delta_p = (th.matmul(p.view(1,-1), Delta)).t().flatten()
                mask_Delta = (Delta_p != 0)
                log_Delta_p = th.zeros_like(Delta_p)
                log_Delta_p[mask_Delta] = th.log(Delta_p[mask_Delta])
                neg_llh = -(th.sum(a * log_p) + th.sum(b * log_Delta_p))
                ell_1_loss = th.nn.functional.l1_loss(p, p_tilde)
                y_term = th.sum(y * p)
                print(f"iter = {it}, objective = {neg_llh}+{ell_1_loss}+{y_term} = {neg_llh + ell_1_loss+y_term}")
            
            if th.max(th.abs(p_next - p)) < eps:
                p = p_next
                break
            p = p_next
        
        assert(th.abs(th.sum(p) - 1) < 1e-6)
        return p
    
class ThetaParam(th.nn.Module):
    def __init__(self, n, T, regularization_func):
        super().__init__()
        self.theta_all_times = Parameter(th.zeros(n*T), requires_grad=True)
        self.n = n
        self.T = T
        self.regularization_func = regularization_func
    
    def forward(self):
        return self.theta_all_times

    def loss(self, p_all_times, y_all_times, connected_components_all_times, d_function, lambd, rho, combined=True):
        theta_all_times = self.forward()
        regularization_loss = lambd * self.regularization_func(theta_all_times, self.n, self.T)
        obj_p_ptilde = 0
        y_term = th.tensor(0., dtype=th.float32)
        
        for t in range(self.T):
            start = self.n * t
            end = self.n * (t+1)
            theta_t = theta_all_times[start:end]
        
            for connected_components in connected_components_all_times[t]:
                temp = th.gather(p_all_times[start:end], 0, connected_components)
                
                p_connected_components = temp / th.sum(temp)
                ptilde_connected_components = torch_softmax(th.gather(theta_t, 0, connected_components))
                
                theta_tilde_t_connected_components = th.gather(theta_t, 0, connected_components)
                theta_tilde_t_connected_components_norm = theta_tilde_t_connected_components - theta_tilde_t_connected_components[0]
                
                theta_t_connected_components = th.log(temp)
                theta_t_connected_components_norm = theta_t_connected_components -  theta_t_connected_components[0]
                
                # obj_p_ptilde += th.nn.functional.mse_loss(theta_t_connected_components_norm, theta_tilde_t_connected_components_norm, size_average=False) * rho # ell2(theta, theta_tilde) loss
                # obj_p_ptilde += th.nn.functional.l1_loss(theta_t_connected_components_norm, theta_tilde_t_connected_components_norm, size_average=False) * rho # ell1(theta, theta_tilde) loss
                # obj_p_ptilde += th.nn.functional.kl_div(p_connected_components, ptilde_connected_components, size_average=False) * rho # KL (ptilde, p) (note the API reverses)
                # obj_p_ptilde += th.nn.functional.kl_div(ptilde_connected_components, p_connected_components, size_average=False) * rho # KL (p, ptilde)
                obj_p_ptilde += th.nn.functional.l1_loss(ptilde_connected_components, p_connected_components, size_average=False) * rho # ell1(p, ptilde) <- This could be the nicest in terms of theoretical justification
                # obj_p_ptilde += th.nn.functional.mse_loss(ptilde_connected_components, p_connected_components, size_average=False) * rho 
                
                
                # TODO: why does leaving out the y-term improve performance?
                # y_term += th.sum(th.gather(y_all_times[start:end], 0, connected_components) * ptilde_connected_components)
        
        if combined:
            return obj_p_ptilde + y_term + regularization_loss
        else:
            return obj_p_ptilde, y_term + regularization_loss
    
    def minimize_lagrangian_wrt_theta(self, optimizer, p_all_times, y_all_times, connected_components_all_times, d_function, lambd, rho,
                                      max_iters=100, eps=1e-6, verbose=False, report=100, reference_theta=None, early_stopping=False):
        
        loss = self.loss(p_all_times, y_all_times, connected_components_all_times, d_function, lambd, rho)
        th.set_printoptions(precision=3)
        np.set_printoptions(precision=3)
        
        if verbose:
            obj_p_ptilde, regularization_loss = self.loss(p_all_times, y_all_times, connected_components_all_times, d_function, lambd, rho, combined=False)
            print(f"init loss = {obj_p_ptilde}+{regularization_loss}={obj_p_ptilde+regularization_loss}")
        
        # cur_theta = self.theta_all_times.detach().numpy()
        cur_loss = loss.detach().numpy()
        cur_theta = self.theta_all_times.detach().numpy()
        if verbose:
            print("Minimizing with respect to theta ... ")
            
        for it in range(max_iters):
            loss.backward(retain_graph=True)
            optimizer.step()

            loss = self.loss(p_all_times, y_all_times, connected_components_all_times, d_function, lambd, rho)
            optimizer.zero_grad()
            
            if verbose and it % report == 0:
                obj_p_ptilde, regularization_loss = self.loss(p_all_times, y_all_times, connected_components_all_times, d_function, lambd, rho, combined=False)
                ell2 = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, self.theta_all_times.detach().numpy())
                print(f"Iteration {it}, Objective = {obj_p_ptilde.detach().numpy()}+{regularization_loss.detach().numpy()}={(obj_p_ptilde+regularization_loss).detach().numpy()}, ell2 = {ell2}")
                
            next_loss = loss.detach().numpy()
            next_theta = self.theta_all_times.detach().numpy()
            if np.abs(next_loss - cur_loss) < eps or np.sum(np.square(next_theta - cur_theta)) or (next_loss > cur_loss and early_stopping):
                break
            cur_loss = next_loss
            cur_theta = next_theta

        if verbose:
            obj_p_ptilde, regularization_loss = self.loss(p_all_times, y_all_times, connected_components_all_times, d_function, lambd, rho, combined=False)
            ell2 = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, self.theta_all_times.detach().numpy())
            print(f"Final iteration {it}, Objective = {obj_p_ptilde.detach().numpy()}+{regularization_loss.detach().numpy()}={(obj_p_ptilde+regularization_loss).detach().numpy()}, ell2 = {ell2}")
    
    def theta_numpy(self):
        return self.theta_all_times.detach().numpy().reshape((self.T, self.n))
            
    def ell2_distance(self, theta_all_times, theta_hat_all_times, connected_components_all_times=None, normalized=True):
        if connected_components_all_times is None:
            connected_components_all_times = [[th.arange(self.n)] for t in range(self.T)]
        
        if theta_hat_all_times.shape != (self.T, self.n):
            theta_hat_all_times = theta_hat_all_times.reshape((self.T, self.n))
        if theta_all_times.shape != (self.T, self.n):
            theta_all_times = theta_all_times.reshape((self.T, self.n))
        ell2 = 0.         
        for t in range(self.T):
            for connected_component in connected_components_all_times[t]:
                theta_hat_t_conn = theta_hat_all_times[t, connected_component]
                theta_t_conn = theta_all_times[t, connected_component]
                if normalized:
                    theta_hat_t_conn -= theta_hat_t_conn[0]
                    theta_t_conn -= theta_t_conn[0]
                ell2 += np.sum(np.square(theta_t_conn- theta_hat_t_conn))
        return ell2


class ItemGPModel(gp.models.ApproximateGP):
    def __init__(self, T, n_inducing_points=None, mean_function="Constant", kernel="Periodic", variational_dist="MeanField"):
        if n_inducing_points is None:
            n_inducing_points = T
        inducing_points = th.tensor(np.linspace(0, T, n_inducing_points), dtype=th.float32)
        
        if variational_dist == "Cholesky":
            variational_distribution = gp.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        elif variational_dist == "MeanField":
            variational_distribution = gp.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        elif variational_dist == "Delta":
            variational_distribution = gp.variational.DeltaVariationalDistribution(inducing_points.size(0))
        else:
            print("Variational distribution not recognized")
            assert(False)
            
        variational_strategy = gp.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ItemGPModel, self).__init__(variational_strategy)
        if mean_function == "Linear":
            self.mean_module = gp.means.LinearMean(input_size=1)
        elif mean_function == "Zero":
            self.mean_module = gp.means.ZeroMean()
        else:
            self.mean_module = gp.means.ConstantMean()
        
        if kernel == "RBF":
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())
        elif kernel == "Cosine":
            self.covar_module =  gp.kernels.ScaleKernel(gp.kernels.CosineKernel())
        elif kernel == "Periodic":
            self.covar_module =  gp.kernels.ScaleKernel(gp.kernels.PeriodicKernel())
        elif kernel == "Matern-1/2" or kernel == "Matern": 
            self.covar_module =  gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=1/2))
        elif kernel == "Matern-3/2": 
            self.covar_module =  gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=3/2))
        elif kernel == "Matern-5/2": 
            self.covar_module =  gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=5/2))
        elif kernel == "Matern-Cosine":
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.MaternKernel()) + gp.kernels.ScaleKernel(gp.kernels.CosineKernel())
        elif kernel == "RBF-Cosine":
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel()) + gp.kernels.ScaleKernel(gp.kernels.CosineKernel())
        elif kernel == "Matern-Periodic":
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.MaternKernel()) + gp.kernels.ScaleKernel(gp.kernels.PeriodicKernel())
        elif kernel == "RBF-Periodic":
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel()) + gp.kernels.ScaleKernel(gp.kernels.PeriodicKernel())
        elif kernel == "SpectralMixture":
            self.covar_module = gp.kernels.SpectralMixtureKernel(5)
        elif kernel == "SpectralDelta":
            self.covar_module = gp.kernels.SpectralDeltaKernel(1)
        elif kernel == "Cylindrical":
            self.covar_module =  gp.kernels.CylindricalKernel(10, gp.kernels.MaternKernel())

        else:
            print("Kernel not recognized")
            assert(False)
        
        self.T = T

    def forward(self, x): # What if we feed it a scalar?
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
        

class GPBackEnd(th.nn.Module):
    def __init__(self, n, T, n_inducing_points=None, mean_function="Constant", kernel="Periodic", variational_dist="MeanField", kernel_args={}):
        super().__init__()
        self.n = n
        self.T = T
        if n_inducing_points is None:
            n_inducing_points = T
        self.n_inducing_points = n_inducing_points

        self.itemGPs = th.nn.ModuleList()
        self.objectives = th.nn.ModuleList()        
        self.likelihood = gp.likelihoods.LaplaceLikelihood()

        for i in range(n):
            self.itemGPs.append(ItemGPModel(T, n_inducing_points, mean_function=mean_function, kernel=kernel, variational_dist=variational_dist))
    
    def forward(self, normalized=True):
        theta_all_times = []
        for i in range(self.n):
            theta_all_times_i = self.itemGPs[i](th.tensor(np.arange(self.T))).mean.detach().numpy()
            theta_all_times.append(theta_all_times_i)
        theta_all_times = np.array(theta_all_times).T
        if normalized:
            theta_all_times -= theta_all_times[:, 0][:, np.newaxis]
        return th.tensor(theta_all_times).flatten()
    
    def loss(self, p_all_times, y_all_times, connected_components_all_times, d_function=None, lambd=1, rho=1, combined=True):        
        est_theta_all_times_by_i = []
        for i in range(self.n):
            est_theta_all_times_by_i.append(self.itemGPs[i](th.tensor(th.arange(self.T), dtype=th.float32)).mean)
        
        obj_p_ptilde = 0
        obj_y_term = th.tensor(0.0, dtype=th.float32)
        
        for t in range(self.T):
            start = self.n * t
            end = self.n * (t+1)
            yt = y_all_times[start:end]
            
            for connected_component in connected_components_all_times[t]:
                temp = th.gather(p_all_times[start:end], 0, connected_component)
                p_connected_components = temp / th.sum(temp)
                theta_tilde_connected_components = th.zeros_like(p_connected_components)
                for idi, i in enumerate(connected_component):
                    theta_tilde_connected_components[idi] = est_theta_all_times_by_i[i][t]
                p_tilde_connected_components = torch_softmax(theta_tilde_connected_components)
                
                obj_p_ptilde += th.nn.functional.l1_loss(p_tilde_connected_components, p_connected_components, size_average=False) * rho
                obj_y_term -= th.sum(yt[connected_component] * p_tilde_connected_components)
                        
        if combined:
            return obj_p_ptilde + obj_y_term
        else:
            return obj_p_ptilde, obj_y_term
                    
    def minimize_lagrangian_wrt_theta(self, optimizer, p_all_times, y_all_times, connected_components_all_times, d_function, lambd, rho,
                                      max_iters=100, eps=1e-3, verbose=False, report=100, reference_theta=None, early_stopping=False):
        
        cur_loss = np.inf
        for it in range(max_iters):
            optimizer.zero_grad()
            loss = self.loss(p_all_times, y_all_times, connected_components_all_times, rho)
            loss.backward()
            optimizer.step()
            
            next_loss = loss.detach().numpy()
            est_theta = self.forward().detach().numpy().reshape((self.T, self.n))
            
            if verbose and it % report == 0:
                ell2 = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, est_theta)
                print(f"Iteration {it}, current loss = {loss.detach().numpy()}, ell2 = {ell2}")
                
            if np.abs(next_loss - cur_loss) < eps:
                break
            cur_loss = next_loss
            
        if verbose:
            est_theta = self.forward().detach().numpy().reshape((self.T, self.n))
            ell2 = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, est_theta)
            print(f"Finally, after {it} iterations, final loss = {loss.detach().numpy()}, ell2 = {ell2}")

    def ell2_distance(self, theta_all_times, theta_hat_all_times, connected_components_all_times=None, normalized=True):
        if connected_components_all_times is None:
            connected_components_all_times = [[th.arange(self.n)] for t in range(self.T)]
        
        if theta_hat_all_times.shape != (self.T, self.n):
            theta_hat_all_times = theta_hat_all_times.reshape((self.T, self.n))
        if theta_all_times.shape != (self.T, self.n):
            theta_all_times = theta_all_times.reshape((self.T, self.n))
            
        ell2 = 0.         
        for t in range(self.T):
            for connected_component in connected_components_all_times[t]:
                theta_hat_t_conn = theta_hat_all_times[t, connected_component]
                theta_t_conn = theta_all_times[t, connected_component]
                if normalized:
                    theta_hat_t_conn -= theta_hat_t_conn[0]
                    theta_t_conn -= theta_t_conn[0]
                ell2 += np.sum(np.square(theta_t_conn- theta_hat_t_conn))
        return ell2
            
   
class DynamicLuce(th.nn.Module):
    def __init__(self, n, T, F, d_function, d_prime, rho=1., lambd=1.):
        super().__init__()
        self.F = F # This should implement a minimize function
        self.n = n
        self.T = T
        self.rho = rho
        self.lambd = lambd
        self.p_all_times = th.ones(n*T, requires_grad=False) * 1./n
        self.d_function = d_function
        self.d_prime = d_prime
        self.y_track = []
        self.theta_track = []
        self.p_track = []
        self.constraint_residual_track = []
        self.objective_tracking = collections.defaultdict(list)
        
        
    def fit(self, rankings_all_times, max_iters=100, eps=1e-4, loss_eps=0.1, verbose=True, report=1, step_size=1.0, lr=0.01, lr_decay=0.99, weight_decay=0.01, max_theta_iters=100, reference_theta=None, include_y=True):
        weaver = RegularizedWeaver()
        y_all_times = th.zeros(self.n*self.T, requires_grad=False)
        
        if type(rankings_all_times) == dict:
            a_all_times, b_all_times, Delta_all_times, connected_components_all_times = rankings_all_times["a_all_times"], \
                rankings_all_times["b_all_times"], rankings_all_times["Delta_all_times"], rankings_all_times["connected_components_all_times"],
        else:
            a_all_times, b_all_times, Delta_all_times, connected_components_all_times = self.construct_a_b_Delta_connected_components_all_times(rankings_all_times)
            
        th.set_printoptions(precision=3)
        np.set_printoptions(precision=3)
            
        
        obj_p_ptilde, y_term_ptilde = self.F.loss(p_all_times=self.p_all_times, y_all_times=y_all_times, connected_components_all_times=connected_components_all_times,
                                d_function=self.d_function, lambd=self.lambd, rho=self.rho, combined=False)
        
        theta_all_times = self.F()
        neg_llh, y_term_p = self.negative_log_likelihood(self.p_all_times, y_all_times, a_all_times, b_all_times, Delta_all_times, connected_components_all_times)
        cur_loss = neg_llh.detach().numpy() + obj_p_ptilde.detach().numpy()

        self.theta_track.append(np.copy(theta_all_times.detach().numpy()))
        self.y_track.append(np.copy(y_all_times.detach().numpy()))
        self.p_track.append(np.copy(self.p_all_times.detach().numpy()))
        
        if verbose:
            ell2 = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, theta_all_times.detach().numpy())
            ell2p = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, np.log(self.p_all_times.detach().numpy().reshape((self.T, self.n))), connected_components_all_times)
            print(f">>> DynamicLuce: Init, Objective = {neg_llh}+{obj_p_ptilde}={neg_llh+obj_p_ptilde}, ell2 (theta) = {ell2}, ell2 (p) = {ell2p}")

        for it in range(max_iters):
            # optimizer = th.optim.Adagrad([self.F.theta_all_times], lr=lr, weight_decay=weight_decay, lr_decay=lr_decay)
            # optimizer = th.optim.SGD([self.F.theta_all_times], lr=lr, momentum=0.9, weight_decay=weight_decay)
            if hasattr(self.F, "n_inducing_points"):
                optimizer = th.optim.SGD(self.F.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            else:
                optimizer = th.optim.SGD([self.F.theta_all_times], lr=lr, momentum=0.9, weight_decay=weight_decay)
            
            theta_all_times = self.F()
            
            # Update p (across all time steps)
            with th.no_grad():
                for t in range(self.T):
                    start_t, end_t = self.n*t, self.n*(t+1)
                    p_truncated_next = weaver.minimize_lagrangian_wrt_p(a=a_all_times[t], b=b_all_times[t], Delta=Delta_all_times[t], 
                                                                           p_init=self.p_all_times[start_t:end_t], p_tilde=torch_softmax(theta_all_times[start_t:end_t]),
                                                                           rho=self.rho, y=y_all_times[start_t:end_t])
                    
                    self.p_all_times[start_t:end_t] = self.p_all_times[start_t:end_t] * (1 - step_size) + step_size * p_truncated_next
                    assert(th.abs(th.sum(self.p_all_times[start_t:end_t]) - 1) < 1e-6)
            
            # Update Theta
            self.F.minimize_lagrangian_wrt_theta(optimizer=optimizer, p_all_times=self.p_all_times, y_all_times=y_all_times,
                    connected_components_all_times=connected_components_all_times, d_function=self.d_function,
                    lambd=self.lambd, rho=self.rho, max_iters=max_theta_iters, reference_theta=reference_theta, verbose=verbose, early_stopping=(it != 0))
            
            # Update y
            if include_y:
                for t in range(self.T):
                    start_t, end_t = self.n * t, self.n * (t+1)
                    for connected_component in connected_components_all_times[t]:
                        y_next = y_all_times[start_t:end_t][connected_component] + \
                            self.rho * (th.nn.functional.normalize(self.p_all_times[start_t:end_t][connected_component], p=1, dim=0)
                                        - torch_softmax(theta_all_times[start_t:end_t][connected_component]))

                        y_all_times[start_t: end_t][connected_component] = y_all_times[start_t: end_t][connected_component] * (1 - step_size) + y_next * step_size            
            
            
            ################################################################
            
            theta_all_times = self.F()
            
            obj_p_ptilde, y_term_ptilde = self.F.loss(p_all_times=self.p_all_times, y_all_times=y_all_times, connected_components_all_times=connected_components_all_times,
                                d_function=self.d_function, lambd=self.lambd, rho=self.rho, combined=False)
            neg_llh, y_term_p = self.negative_log_likelihood(self.p_all_times, y_all_times, a_all_times, b_all_times, Delta_all_times, connected_components_all_times)
            next_loss = neg_llh.detach().numpy() + obj_p_ptilde.detach().numpy()
            
            if verbose and it % report== 0:
                ell2 = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, theta_all_times.detach().numpy())
                ell2p = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, np.log(self.p_all_times.detach().numpy().reshape((self.T, self.n))), connected_components_all_times)
                print(f">>> DynamicLuce: Iteration {it}, Objective = {neg_llh}+{obj_p_ptilde}={neg_llh+obj_p_ptilde}, ell2 (theta) = {ell2}, ell2 (p) = {ell2p}")
                print()
            
            if (len(self.theta_track) > 1 and (np.sum(np.square(self.theta_track[-1] - self.theta_track[-2])) < eps)) or np.abs(next_loss - cur_loss) < loss_eps:
                break
            
            cur_loss = next_loss
            
            self.theta_track.append(np.copy(theta_all_times.detach().numpy()))
            self.y_track.append(np.copy(y_all_times.detach().numpy()))
            self.p_track.append(np.copy(self.p_all_times.detach().numpy()))
            
            self.objective_tracking["obj_p_ptilde"].append(obj_p_ptilde.detach().numpy())
            self.objective_tracking["neg_llh"].append(neg_llh.detach().numpy())
            self.objective_tracking["y_term_ptilde"].append(y_term_ptilde.detach().numpy())
            self.objective_tracking["y_term_p"].append(y_term_p.detach().numpy())
            
            lr *= lr_decay
        
        if verbose:
            ell2 = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, theta_all_times.detach().numpy())
            ell2p = "N/A" if reference_theta is None else self.ell2_distance(reference_theta, np.log(self.p_all_times.detach().numpy().reshape((self.T, self.n))), connected_components_all_times)
            print(f">>> Dynamic Luce: Final iteration {it}, Objective = {neg_llh}+{obj_p_ptilde}={neg_llh+obj_p_ptilde}, ell2 (theta) = {ell2}, ell2 (p) = {ell2p}")
            print()
            
    def negative_log_likelihood(self, p_all_times, y_all_times, a_all_times, b_all_times, Delta_all_times, connected_components_all_times):
        neg_llh = 0.
        y_term = 0.
        for t in range(self.T):
            start_t, end_t = self.n * t, self.n * (t+1)
            p_t = p_all_times[start_t:end_t]
            a_t = a_all_times[t]
            b_t = b_all_times[t]
            Delta_t = Delta_all_times[t]
            y_t = y_all_times[t]
            
            log_pt = th.zeros_like(p_t)
            mask = (p_t != 0)
            log_pt[mask] = th.log(p_t[mask])
            
            Delta_p = th.matmul(p_t.view(1,-1), Delta_t).t().flatten()
            log_Delta_p = th.zeros_like(Delta_p)
            mask = (Delta_p != 0)
            log_Delta_p[mask] = th.log(Delta_p[mask])
            neg_llh -= (th.sum(a_t * log_pt) + th.sum(b_t * log_Delta_p))
            
            y_term += th.sum(y_t * p_t)
            
        return neg_llh, y_term
            
    def ell2_distance(self, theta_all_times, theta_hat_all_times, connected_components_all_times=None, normalized=True):
        if connected_components_all_times is None:
            connected_components_all_times = [[th.arange(self.n)] for t in range(self.T)]
        
        if theta_hat_all_times.shape != (self.T, self.n):
            theta_hat_all_times = theta_hat_all_times.reshape((self.T, self.n))
        if theta_all_times.shape != (self.T, self.n):
            theta_all_times = theta_all_times.reshape((self.T, self.n))
        ell2 = 0.         
        for t in range(self.T):
            for connected_component in connected_components_all_times[t]:
                theta_hat_t_conn = theta_hat_all_times[t, connected_component]
                theta_t_conn = theta_all_times[t, connected_component]
                if normalized:
                    theta_hat_t_conn -= theta_hat_t_conn[0]
                    theta_t_conn -= theta_t_conn[0]
                ell2 += np.sum(np.square(theta_t_conn- theta_hat_t_conn))
        return ell2
    
    def get_p_all_times(self):
        return self.p_all_times.detach().numpy().reshape((self.T, self.n))
    
    def get_theta_all_times(self, normalized=True):
        theta_all_times = self.F().detach().numpy()
        theta_all_times = theta_all_times.reshape((self.T,self.n))
        if normalized:
            theta_all_times = theta_all_times - theta_all_times[:, 0][:, np.newaxis]
        return theta_all_times
        
    def construct_a_b_Delta_connected_components_all_times(self, rankings_all_times, reduced=True):
        a_all_times = []
        b_all_times = []
        Delta_all_times = []
        connected_components_all_times = []
        
        for rankings in rankings_all_times:
            at, bt, Delta_t, connected_components_t = self.compute_a_b_Delta_connected_components(rankings, reduced)
            a_all_times.append(at)
            b_all_times.append(bt)
            Delta_all_times.append(Delta_t)
            connected_components_all_times.append(connected_components_t)
        
        return a_all_times, b_all_times, Delta_all_times, connected_components_all_times
    
    def compute_a_b_Delta_connected_components(self, ranked_data, reduced=True):
        from rums.algorithms.utils import find_connected_components_from_menus
        import collections
        
        a = th.zeros(self.n)
        b = []
        Delta = []
        all_menus = set()
        
        for _, rank in enumerate(ranked_data):
            all_menus.add(tuple(list(np.sort(rank))))
            for idi in range(len(rank)-1):
                a[rank[idi]] += 1
                delta_l = np.zeros((self.n,))
                for idj in range(idi, len(rank)):
                    delta_l[rank[idj]] = 1
                Delta.append(delta_l)
                b.append(-1)
                
        if reduced: # Combine repeated rows to save computation
            def get_key_from_row(row):
                return ''.join(str(int(s)) for s in row)
            def get_row_from_key(key):
                return [float(s) for s in key]
            
            unique_delta_rows = collections.defaultdict(int)
            for delta_row in Delta:
                unique_delta_rows[get_key_from_row(delta_row)] += 1
            
            Delta = []
            b = []
            for key, count in unique_delta_rows.items():
                Delta.append(get_row_from_key(key))
                b.append(-count)
                
        Delta = th.tensor(Delta, dtype=th.float32).T
        b = th.tensor(b)
        connected_components = find_connected_components_from_menus(all_menus)
        connected_components = [th.tensor(s) for s in connected_components]
        return a, b, Delta, connected_components
