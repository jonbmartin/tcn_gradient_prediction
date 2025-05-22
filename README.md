# TCN Gradient Trajectory Error Prediction
This is a code repository accompanying the MRM paper "Improved Image Reconstruction and Diffusion Parameter Estimation Using a Temporal Convolutional Network Model of Gradient Trajectory Errors"

| Characteristic        | Value                      |
|-----------------------|---------------------------|
| **Training Dataset**  | Purpose: To train the model |
|                       | Size: Largest portion     |
|                       | Data Leakage Risks: High, if overfitting occurs |
|                       | Use of Labels: Required    |
|                       | Feedback Loop: Yes         |
|                       | Iteration: Multiple iterations |
|                       | Role in Model: Primary source of learning |
|                       | Distribution: Ideally representative |

| **Validation Dataset**| Purpose: To tune model parameters |
|                       | Size: Smaller portion       |
|                       | Data Leakage Risks: Moderate, if not carefully managed |
|                       | Use of Labels: Required      |
|                       | Feedback Loop: Yes (during hyperparameter tuning) |
|                       | Iteration: Limited iterations |
|                       | Role in Model: Model tuning and selection |
|                       | Distribution: Should mimic training distribution |

| **Test Dataset**      | Purpose: To evaluate model performance |
|                       | Size: Medium to small portion |
|                       | Data Leakage Risks: Low, should be independent |
|                       | Use of Labels: Required      |
|                       | Feedback Loop: No            |
|                       | Iteration: No iterations      |
|                       | Role in Model: Final evaluation |
|                       | Distribution: Should mimic real-world distribution |
