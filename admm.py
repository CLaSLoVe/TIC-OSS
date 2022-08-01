import numpy as np
import math


class ADMMSolver:
	def __init__(self, lamb, num_stacked, size_blocks, rho, S, rho_update_func=None):
		self.lamb = lamb
		self.numBlocks = num_stacked
		self.sizeBlocks = size_blocks
		probSize = num_stacked * size_blocks
		self.length = int(probSize * (probSize + 1) / 2)
		self.x = np.zeros(self.length)
		self.z = np.zeros(self.length)
		self.u = np.zeros(self.length)
		self.rho = float(rho)
		self.S = S
		self.status = 'initialized'
		self.rho_update_func = rho_update_func
	
	def ij2symmetric(self, i, j, size):
		return (size * (size + 1)) / 2 - (size - i) * ((size - i + 1)) / 2 + j - i
	
	def upper2Full(self, a):
		n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
		A = np.zeros([n, n])
		A[np.triu_indices(n)] = a
		temp = A.diagonal()
		A = (A + A.T) - np.diag(temp)
		return A
	
	def Prox_logdet(self, S, A, eta):
		d, q = np.linalg.eigh(eta * A - S)
		q = np.matrix(q)
		X_var = (1 / (2 * float(eta))) * q * (
			np.diag(d + np.sqrt(np.square(d) + (4 * eta) * np.ones(d.shape)))) * q.T
		x_var = X_var[np.triu_indices(S.shape[1])]  # extract upper triangular part as update variable
		return np.matrix(x_var).T
	
	def ADMM_x(self):
		a = self.z - self.u
		A = self.upper2Full(a)
		eta = self.rho
		x_update = self.Prox_logdet(self.S, A, eta)
		self.x = np.array(x_update).T.reshape(-1)
	
	def ADMM_z(self, index_penalty=1):
		a = self.x + self.u
		probSize = self.numBlocks * self.sizeBlocks
		z_update = np.zeros(self.length)
		
		# TODO: can we parallelize these?
		for i in range(self.numBlocks):
			elems = self.numBlocks if i == 0 else (2 * self.numBlocks - 2 * i) / 2  # i=0 is diagonal
			for j in range(self.sizeBlocks):
				startPoint = j if i == 0 else 0
				for k in range(startPoint, self.sizeBlocks):
					locList = [((l + i) * self.sizeBlocks + j, l * self.sizeBlocks + k) for l in range(int(elems))]
					if i == 0:
						lamSum = sum(self.lamb[loc1, loc2] for (loc1, loc2) in locList)
						indices = [self.ij2symmetric(loc1, loc2, probSize) for (loc1, loc2) in locList]
					else:
						lamSum = sum(self.lamb[loc2, loc1] for (loc1, loc2) in locList)
						indices = [self.ij2symmetric(loc2, loc1, probSize) for (loc1, loc2) in locList]
					pointSum = sum(a[int(index)] for index in indices)
					rhoPointSum = self.rho * pointSum
					
					# Calculate soft threshold
					ans = 0
					# If answer is positive
					if rhoPointSum > lamSum:
						ans = max((rhoPointSum - lamSum) / (self.rho * elems), 0)
					elif rhoPointSum < -1 * lamSum:
						ans = min((rhoPointSum + lamSum) / (self.rho * elems), 0)
					
					for index in indices:
						z_update[int(index)] = ans
		self.z = z_update
	
	def ADMM_u(self):
		u_update = self.u + self.x - self.z
		self.u = u_update
	
	# Returns True if convergence criteria have been satisfied
	# eps_abs = eps_rel = 0.01
	# r = x - z
	# s = rho * (z - z_old)
	# e_pri = sqrt(length) * e_abs + e_rel * max(||x||, ||z||)
	# e_dual = sqrt(length) * e_abs + e_rel * ||rho * u||
	# Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
	# Returns (boolean shouldStop, primal residual value, primal threshold,
	#          dual residual value, dual threshold)
	def CheckConvergence(self, z_old, e_abs, e_rel, verbose):
		norm = np.linalg.norm
		r = self.x - self.z
		s = self.rho * (self.z - z_old)
		# Primal and dual thresholds. Add .0001 to prevent the case of 0.
		e_pri = math.sqrt(self.length) * e_abs + e_rel * max(norm(self.x), norm(self.z)) + .0001
		e_dual = math.sqrt(self.length) * e_abs + e_rel * norm(self.rho * self.u) + .0001
		# Primal and dual residuals
		res_pri = norm(r)
		res_dual = norm(s)
		if verbose:
			# Debugging information to print(convergence criteria values)
			print('  r:', res_pri)
			print('  e_pri:', e_pri)
			print('  s:', res_dual)
			print('  e_dual:', e_dual)
		stop = (res_pri <= e_pri) and (res_dual <= e_dual)
		return (stop, res_pri, e_pri, res_dual, e_dual)
	
	# solve
	def __call__(self, maxIters, eps_abs, eps_rel, verbose):
		num_iterations = 0
		self.status = 'Incomplete: max iterations reached'
		for i in range(maxIters):
			z_old = np.copy(self.z)
			self.ADMM_x()
			self.ADMM_z()
			self.ADMM_u()
			if i != 0:
				stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(z_old, eps_abs, eps_rel, verbose)
				if stop:
					self.status = 'Optimal'
					break
				new_rho = self.rho
				if self.rho_update_func:
					new_rho = rho_update_func(self.rho, res_pri, e_pri, res_dual, e_dual)
				scale = self.rho / new_rho
				rho = new_rho
				self.u = scale * self.u
			if verbose:
				# Debugging information prints current iteration #
				print('Iteration %d' % i)
		return self.x


def upperToFull(a, eps=0):
	ind = (a < eps) & (a > -eps)
	a[ind] = 0
	n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
	A = np.zeros([n, n])
	A[np.triu_indices(n)] = a
	temp = A.diagonal()
	A = np.asarray((A + A.T) - np.diag(temp))
	return A


if __name__ == '__main__':
	import numpy as np
	m = 100
	n = 10
	w = 1
	lambda_parameter = 11e-2
	window_size = w
	size_blocks = n
	probSize = window_size * size_blocks
	lamb = np.zeros((probSize, probSize)) + lambda_parameter
	D_train = np.loadtxt('../example_data.txt', delimiter=',')[:m, :]
	S = np.cov(np.transpose(D_train))
	admm = ADMMSolver(lamb, window_size, size_blocks, 1, S)
	admm(1000, 1e-6, 1e-6, False)
	S_est = upperToFull(admm.x, 1e-6)
	X2 = S_est
	u, _ = np.linalg.eig(S_est)
	cov_out = np.linalg.inv(X2)
