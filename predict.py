import numpy as np
import torch

def infer_score(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, score_lambda):
    scores = calculate_anomaly_score(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, score_lambda)
    return scores

def infer_label(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, score_lambda, train_data_loader, test_data_loader, anomaly_ratio):
    train_scores = calculate_anomaly_score(model, train_data_loader, time_anomaly_criterion, freq_anomaly_criterion, score_lambda)
    test_scores = calculate_anomaly_score(model, test_data_loader, time_anomaly_criterion, freq_anomaly_criterion, score_lambda)
    combined_scores = np.concatenate([train_scores, test_scores], axis=0)
    thre_scores = calculate_anomaly_score(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, score_lambda)
    preds = {}
    for ratio in anomaly_ratio:
        threshold = np.percentile(combined_scores, 100 - ratio)
        preds[ratio] = (thre_scores > threshold).astype(int)
    # print(preds, thre_scores)
    return preds, thre_scores

def calculate_anomaly_score(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, score_lambda):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    attens_energy = []
    all_alphas = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(thre_loader):
            batch_x = batch_x.float().to(device)
            # reconstruction
            outputs = model(batch_x)
            output = outputs["z"]
            alpha = outputs["TFAD_alpha"]
            all_alphas.append(alpha.detach().cpu().numpy())
            # criterion
            time_score = torch.mean(time_anomaly_criterion(batch_x, output), dim=-1)
            freq_score = torch.mean(freq_anomaly_criterion(batch_x, output), dim=-1)
            score = (
                (time_score + score_lambda * freq_score).detach().cpu().numpy()
            )
            attens_energy.append(score)
    
    # Calculate and print average alpha
    avg_alpha = np.mean(np.concatenate(all_alphas, axis=0), axis=0)
    print(f"\n[TFAD Info] Average Feature Importance (Alpha):")
    features = ["Residual", "Context Residual", "Cyclic", "Context Cyclic"]
    for feat, val in zip(features, avg_alpha):
        print(f"  - {feat}: {val:.4f}")
    print("-" * 40)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    scores = np.array(attens_energy)
    return scores