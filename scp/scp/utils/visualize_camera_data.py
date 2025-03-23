import matplotlib.pyplot as plt
def visualize_camera_data(rgb=None, mask=None, points=None, figsize=(15,5)):
    """
    Visualize camera data including RGB image, segmentation mask, and point cloud.
    
    Args:
        rgb (ndarray, optional): RGB image array
        mask (ndarray, optional): Segmentation mask array
        points (ndarray, optional): Point cloud array
        figsize (tuple): Figure size for the plot
    """
    num_plots = sum(x is not None for x in [rgb, mask, points])
    if num_plots == 0:
        return
        
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    if rgb is not None:
        axes[plot_idx].imshow(rgb)
        axes[plot_idx].set_title('RGB Image')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
    if mask is not None:
        axes[plot_idx].imshow(mask)
        axes[plot_idx].set_title('Segmentation Mask')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
    if points is not None:
        # Assuming points is a Nx3 array
        axes[plot_idx].scatter(points[:,0], points[:,1], s=1)
        axes[plot_idx].set_title('Point Cloud (Top View)')
        axes[plot_idx].axis('equal')
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()