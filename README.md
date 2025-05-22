# TCN Gradient Trajectory Error Prediction
This is a code repository accompanying the MRM paper "Improved Image Reconstruction and Diffusion Parameter Estimation Using a Temporal Convolutional Network Model of Gradient Trajectory Errors"

| Characteristic        | Training Dataset          | Validation Dataset        | Test Dataset             |
|-----------------------|---------------------------|---------------------------|--------------------------|
| Purpose               | To train the model        | To tune model parameters   | To evaluate model performance |
| Size                  | Largest portion           | Smaller portion            | Medium to small portion  |
| Data Leakage Risks    | High, if overfitting occurs| Moderate, if not carefully managed| Low, should be independent |
| Use of Labels         | Required                  | Required                   | Required                 |
| Feedback Loop         | Yes                       | Yes (during hyperparameter tuning) | No                      |
| Iteration             | Multiple iterations        | Limited iterations         | No iterations             |
| Role in Model        | Primary source of learning | Model tuning and selection | Final evaluation         |
| Distribution          | Ideally representative     | Should mimic training distribution | Should mimic real-world distribution |
