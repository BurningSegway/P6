import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(covariances, means, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    covariances: np.ndarray
        An upper triangular matrix with covariances.
    
    means: list or np.ndarray
        The center of the ellipse, i.e. the means of the distribution.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    # if x.size != y.size:
    #     raise ValueError("x and y must be the same size")

    # covariances = np.covariances(x, y)
    pearson = covariances[0, 1]/np.sqrt(covariances[0, 0] * covariances[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(covariances[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(covariances[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(means[0], means[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

fig, ax = plt.subplots(1,1)
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.plot(5,5,"-o")
confidence_ellipse(np.array([[2., .2],
                             [1., 4.]]), [1.2, 3.4], ax, edgecolor="black")
plt.show()