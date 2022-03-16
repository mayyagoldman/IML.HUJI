from IMLearn.learners import UnivariateGaussian, MultivariateGaussian, gaussian_estimators
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    n = 1000
    mu = 10
    var = 1
    X = np.random.normal(mu, var, size=n)
    estimator = gaussian_estimators.UnivariateGaussian().fit(X)
    print("(", estimator.mu_, estimator.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(np.int_)
    estimated_mean = []
    for m in ms:
        _X = X[:m]
        estimated_mean.append(abs(gaussian_estimators.UnivariateGaussian().fit(_X).mu_ - mu))  # emprical error
    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(3.1.2) Absolute distance between the estimated and empirical expectation against sample size}$",
                  xaxis_title="$\\text{number of samples}$",
                  yaxis_title="r$|\hat\mu - \mu|$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=X, y=estimator.pdf(X), mode='markers', name='$\\text{estimated PDF}$'),
               go.Scatter(x=X, y=np.norm.pdf(X, mu, var), mode='markers', name='$\\text{empirical PDF}$')],
              layout=go.Layout(title="$\\text{(3.1.3) Estimated vs Empirical PDF }$",
                               xaxis_title="$\\text{ sample values}$",
                               yaxis_title="$\\text{PDF}$",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu_mult = [0, 0, 4, 0]
    sigma_mult = \
        [[1, 0.2, 0, 0.5],
         [0.2, 2, 0, 0],
         [0, 0, 1, 0],
         [0.5, 0, 0, 1]]
    X_mult = np.random.multivariate_normal(mu_mult, sigma_mult, n)
    estimator_mult = gaussian_estimators.MultivariateGaussian().fit(X_mult)
    print(estimator_mult.mu_)
    print(estimator_mult.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    L = [estimator_mult.log_likelihood(np.array([i, 0, j, 0]), sigma_mult, X_mult) for i in f1 for j in f3]
    L = np.array(L).reshape((200, 200))
    fig = go.Figure(data=go.Heatmap(z=L.tolist(), x=f1, y=f3))
    fig.show()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
