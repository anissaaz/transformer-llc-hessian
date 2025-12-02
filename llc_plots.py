import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_llc_and_loss(folder_path):
    # Set the visual style
    sns.set_theme(style="whitegrid")
    
    # 1. Plot all Layers in one figure
    # Pattern to match layer files
    layer_pattern = os.path.join(folder_path, 'llc_layer*_attention_query_key_value_weight.csv')
    layer_files = sorted(glob.glob(layer_pattern))
    
    if layer_files:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Iterate through each layer file and plot
        for file_path in layer_files:
            try:
                df = pd.read_csv(file_path)
                
                # Extract a readable label (e.g., "layer0") from filename
                filename = os.path.basename(file_path)
                # Split by '_' and take the second part "layerX"
                layer_label = filename.split('_')[1]
                
                # Plot LLC
                sns.lineplot(ax=axes[0], x='step', y='llc', data=df, label=layer_label)
                # Plot Loss
                sns.lineplot(ax=axes[1], x='step', y='loss', data=df, label=layer_label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # formatting the plots
        axes[0].set_title('Layers: LLC over Steps')
        axes[0].set_xscale('log') # Log scale for steps
        axes[0].set_ylabel('LLC')
        
        axes[1].set_title('Layers: Loss over Steps')
        axes[1].set_xscale('log')
        axes[1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('combined_layers_plot.png')
        print("Saved combined_layers_plot.png")
        plt.show()
    else:
        print("No layer files found.")

    # 2. Plot remaining CSVs separately
    # Get all csvs and filter out the layer ones we just plotted
    all_csvs = glob.glob(os.path.join(folder_path, '*.csv'))
    other_files = [f for f in all_csvs if f not in layer_files]
    
    for file_path in other_files:
        try:
            df = pd.read_csv(file_path)
            name = os.path.basename(file_path).replace('.csv', '')
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot LLC
            sns.lineplot(ax=axes[0], x='step', y='llc', data=df, color='blue')
            axes[0].set_title(f'{name}: LLC')
            axes[0].set_xscale('log')
            
            # Plot Loss
            sns.lineplot(ax=axes[1], x='step', y='loss', data=df, color='orange')
            axes[1].set_title(f'{name}: Loss')
            axes[1].set_xscale('log')
            
            plt.suptitle(name, fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            output_name = f'{name}_plot.png'
            plt.savefig(output_name)
            print(f"Saved {output_name}")
            plt.show()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    folder_name = 'llc-bs64-steps400'
    if os.path.exists(folder_name):
        plot_llc_and_loss(folder_name)
    else:
        print(f"Folder '{folder_name}' not found. Please check the path.")