import torch
import numpy as np
import shap

class QuantLSTMExplainer:
    def __init__(self, model, background_tensor=None):
        """
        Initializes the SHAP DeepExplainer for the PyTorch LSTM.
        
        Args:
            model (nn.Module): The trained PyTorch LSTM model.
            background_tensor (tuple, optional): A background dataset for SHAP to compute expected values.
                Since our model takes (X_seq, X_macro), this should be a tuple of two tensors.
        """
        self.model = model
        
        if self.model is not None:
            self.model.eval()
            
        if background_tensor is None or self.model is None:
            raise ValueError("Strict Mode: Background tensor and Model required for real SHAP values. NO MOCK ALLOWED.")
        else:
            try:
                self.explainer = shap.DeepExplainer(self.model, background_tensor)
                print("[Explainer] SHAP DeepExplainer initialized successfully.")
            except Exception as e:
                print(f"[Explainer] Error initializing DeepExplainer: {e}. Running in Mock mode.")
                self.explainer = None
            
    def get_feature_contributions(self, sequence_tensor, macro_tensor, feature_names: list, class_idx=2):
        """
        Extracts SHAP values and returns the top driving features.
        If the true explainer isn't wired yet, returns mock insights for the pipeline.
        
        Returns:
            top_pos_features (str), top_neg_features (str)
        """
        if self.explainer is None or self.model is None:
            raise ValueError("Explainer or Model is None. Cannot compute SHAP values.")
            
        # REAL SHAP IMPLEMENTATION
        try:
            # SHAP requires requires_grad=True
            sequence_tensor.requires_grad = True
            macro_tensor.requires_grad = True
            
            # Calculate SHAP values. Output shape matches input tuple.
            shap_values = self.explainer.shap_values((sequence_tensor, macro_tensor))
            
            # shap_values is a list of arrays (one for each output class).
            # We care about the target class (e.g., class_idx=2 for UP)
            shap_class_values = shap_values[class_idx]
            
            # shap_class_values[0] is the sequence features shape: (batch_size, seq_len, num_features)
            shap_seq = shap_class_values[0] 
            
            # Average across the sequence dimension (dim=1) to get overall feature importance
            avg_shap_seq = np.mean(shap_seq, axis=1) # Shape: (batch_size, num_features)
            
            # Get the feature importances for the first sample in the batch
            importances = avg_shap_seq[0]
            
            # Sort indices
            sorted_indices = np.argsort(importances)
            
            # Top 2 Positive (largest SHAP values)
            top_pos_idx = sorted_indices[-2:]
            top_pos_features = f"🔼 {feature_names[top_pos_idx[1]]} (+{importances[top_pos_idx[1]]:.3f}), {feature_names[top_pos_idx[0]]} (+{importances[top_pos_idx[0]]:.3f})"
            
            # Top 1 Negative (smallest SHAP values, i.e., most negative)
            top_neg_idx = sorted_indices[0]
            top_neg_features = f"🔽 {feature_names[top_neg_idx]} ({importances[top_neg_idx]:.3f})"
            
            return top_pos_features, top_neg_features
            
        except Exception as e:
            print(f"[Explainer] Error calculating SHAP values: {e}. Falling back to default.")
            return "N/A (Explanation unavailable)", "N/A (Explanation unavailable)"
