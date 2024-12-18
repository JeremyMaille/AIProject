import os
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import gc
from .config import ModelConfig, LEARNING_RATES, DROPOUT_RATES, LAYER_SIZES, BATCH_SIZES
from .model_trainer import ModelTrainer

def train_model_process(args):
    """
    Process-safe training function that trains a model and saves it if it meets the AUC threshold.
    """
    try:
        config, X_train, y_train, X_test, y_test, model_id = args
        print(f"\nStarting training for Model {model_id} with config: {config}")

        trainer = ModelTrainer(config)
        model, history = trainer.train_single_model(
            X_train, y_train, X_test, y_test, 
            epochs=50  # Adjust epochs as needed
        )
        metrics = model.evaluate(X_test, y_test, verbose=0)
        print(f"Model {model_id} completed: Loss={metrics[0]:.4f}, Accuracy={metrics[1]:.4f}, AUC={metrics[2]:.4f}")

        # Define AUC threshold
        auc_threshold = 0.55  # Lowered threshold for debugging
        if metrics[2] > auc_threshold:
            # Ensure the 'saved_models' directory exists
            os.makedirs('saved_models', exist_ok=True)
            model_path = os.path.join('saved_models', f"model_{model_id}.h5")
            model.save(model_path)
            print(f"Model {model_id} saved to {model_path}")

            result = {
                'model_id': model_id,
                'metrics': dict(zip(['loss', 'accuracy', 'auc'], metrics)),
                'config': config.to_dict(),
                'model_path': model_path  # Store the path instead of the model object
            }
            del model, history
            gc.collect()
            return result
        else:
            print(f"Model {model_id} did not meet the AUC threshold (AUC={metrics[2]:.4f})")
            del model, history
            gc.collect()
            return None

    except Exception as e:
        print(f"Error in Model {model_id}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

class ParallelTrainer:
    def __init__(self):
        self.configs = self._generate_configs()
        total_cores = cpu_count()
        self.num_cores = max(1, total_cores - 2)  # Leave 2 cores free
        # Optional: Cap the number of cores to prevent overuse on systems with many cores
        if self.num_cores > 8:
            self.num_cores = 8
        print(f"Generated {len(self.configs)} configurations")
        print(f"Using {self.num_cores} CPU cores out of {total_cores} available")
        self.feature_importances = None

    def _generate_configs(self):
        """
        Generate all combinations of hyperparameters.
        """
        configs = []
        model_id = 0
        for lr in LEARNING_RATES:
            for dr in DROPOUT_RATES:
                for ls in LAYER_SIZES:
                    for bs in BATCH_SIZES:
                        configs.append((
                            ModelConfig(lr, dr, ls, bs),
                            model_id
                        ))
                        model_id += 1
        return configs

    def train_parallel(self, X_train, y_train, X_test, y_test):
        """
        Train models in parallel using multiprocessing.
        """
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        y_train_np = y_train.values
        y_test_np = y_test.values

        all_results = []
        total_configs = len(self.configs)

        print(f"\nStarting training of {total_configs} model configurations")

        with tqdm(total=total_configs, desc="Training models") as pbar:
            with Pool(processes=self.num_cores) as pool:
                pool_args = [
                    (config, X_train_np, y_train_np, X_test_np, y_test_np, model_id)
                    for config, model_id in self.configs
                ]
                results = pool.map(train_model_process, pool_args)
                for res in results:
                    if res:
                        all_results.append(res)
                    pbar.update(1)
                gc.collect()

        print(f"\nTraining completed: {len(all_results)} successful models out of {total_configs}")
        return all_results

    def feature_importance(self, feature_names):
        """
        Implement feature importance analysis using SHAP.
        """
        import pandas as pd
        import shap

        # Check if there are any successful models
        if not self.feature_importances:
            print("Feature importance analysis requires trained models. Ensure models are trained and 'feature_importances' is set.")
            return pd.Series()

        # Example using SHAP with the best model
        best_model_info = max(self.feature_importances, key=lambda x: x['metrics']['auc'])
        model = tf.keras.models.load_model(best_model_info['model_path'])
        X_sample = X_train[:100].values.astype(np.float32)  # Use a sample for efficiency

        explainer = shap.DeepExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        shap_plot_path = os.path.join('results', 'feature_importance.png')
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"Feature importance plot saved to {shap_plot_path}")
        return pd.Series()
