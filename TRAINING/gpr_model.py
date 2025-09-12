import gpytorch

class GPRegressionModel(gpytorch.models.ExactGP):
    """
    A Gaussian Process Regression (GPR) model using GPyTorch.
    
    This class defines a GPR model with a flexible kernel.
    
    Args:
        train_x (torch.Tensor): Training input data.
        train_y (torch.Tensor): Training target data.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood for the GP model.
        kernel (gpytorch.kernels.Kernel, optional): The kernel to use in the GP model. Defaults to Matérn kernel.
    """
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Set default kernel to Matérn if none is provided
        if kernel is None:
            kernel = gpytorch.kernels.MaternKernel(nu=3./2)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
    
    def forward(self, x):
        """
        Forward pass through the GP model.
        
        Args:
            x (torch.Tensor): Input data.
        
        Returns:
            gpytorch.distributions.MultivariateNormal: The output distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultitaskGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=3./2), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)