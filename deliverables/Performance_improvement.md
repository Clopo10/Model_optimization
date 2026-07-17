# Machine Learning Model Performance Review
**Project:** CIFAR-10 Image Classification
**Hardware:** Nvidia GTX 1050 Ti (4GB VRAM)
**Dataset Name:** CIFAR-10
**Evaluation Metric:** Validation Accuracy & Validation Loss
**Overfitting Check:** Monitored via Early Stopping

---

## 1. Phase One: Baseline Model Validation
The initial phase focused on establishing a working data pipeline and proving the underlying hardware configuration was stable. 

*   **Architecture:** Simple Custom CNN (Un-fused Conv2D and ReLU layers to prevent CuDNN 5003 errors on Pascal-architecture GPUs).
*   **Training Duration:** 3 Epochs.
*   **Key Findings:**
    *   The model bypassed the random-guess threshold (10%) and successfully learned baseline spatial features.
    *   Cross-validation revealed a highly stable architecture, proving the network was generalizing rather than memorizing.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Validation Accuracy** | 61.08% | Expected baseline for 3 epochs. |
| **Variance** | +/- 0.72% | Extremely tight window; indicates high stability. |

---

## 2. Phase Two: Hyperparameter Tuning
To push the model beyond the baseline, three distinct search algorithms were deployed to find the optimal layer widths, dropout rates, and learning rates. 

*   **Constraints:** To maintain a fair test and respect hardware limits, all algorithms were capped at 10 trials, running for 4 epochs per trial.
*   **Key Findings:** 
    *   Grid Search severely underperformed due to its rigid, incremental nature.
    *   Random Search performed slightly better through sheer probability.
    *   Bayesian Optimization successfully adapted to the search space, mapping out the highest-yielding parameter zones.

| Search Algorithm | Epoch 4 Accuracy | Strategy |
| :--- | :--- | :--- |
| **Grid Search** | 68.65% | Tested rigid, bottom-tier parameter combinations. |
| **Random Search** | 69.30% | Relied entirely on blind, randomized testing. |
| **Bayesian Optimization** | **72.23%** | Learned from previous trials to predict the best parameters. |

---

## 3. Phase Three: The Overfitting Event
The winning architecture from the Bayesian search was built and trained for a full 20 epochs to test its absolute maximum capacity. 

*   **The Sweet Spot (Epochs 0 to 6):** The model generalized perfectly, reaching its highest validation accuracy and lowest validation loss.
*   **The Overfitting Zone (Epochs 7 to 20):** The model exhausted its ability to learn broad concepts and began directly memorizing the training dataset. 
*   **Key Findings:** While training accuracy artificially climbed toward 95%, validation loss climbed in a classic U-shape (peaking at 1.35), proving the extended training duration was actively harming the model's ability to classify unseen images.

---

## 4. Phase Four: The Final Optimized Model
To capture the model at its absolute peak intelligence, an automated `EarlyStopping` callback was integrated into the training loop.

*   **Trigger Condition:** The callback monitored Validation Loss with a patience of 3 epochs. 
*   **The Execution:**
    *   The model hit its peak at Epoch 6.
    *   The callback observed the loss rising during Epochs 7, 8, and 9.
    *   Training was immediately terminated at Epoch 9 to save GPU compute time.
*   **The Rollback:** The damaged, overfitted weights from Epoch 9 were automatically discarded, and the pristine weights from Epoch 6 were permanently restored to the model.

> **Final Conclusion:** By combining hardware-safe data pipelines, Bayesian hyperparameter tuning, and strict early-stopping guardrails, the final architecture successfully maximized the CIFAR-10 validation accuracy while entirely neutralizing the threat of overfitting.