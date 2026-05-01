# Privacy Demo Testing Guide

Use this guide to test and present the privacy proof.

## 1. Run the comparison demo only

Open PowerShell:

```powershell
cd C:\Users\Manas\Documents\Playground\partner_repo_inspect
python main.py --mode privacy_demo
```

Expected terminal result:

```text
Scenario        : Centralized Raw Data
TCP Port        : 5001
Payload Keys    : ['client_id', 'raw_sensor_data', 'activity_labels', 'subject_ids', 'description']
Privacy Risk    : True

Scenario        : Federated Model Update
TCP Port        : 5000
Payload Keys    : ['client_id', 'weights', 'num_samples', 'description']
Privacy Risk    : False
```

Interpretation:

- Centralized demo sends raw HAR sensor data and labels to the server.
- Federated demo sends only model weights and sample count.
- This demonstrates raw-data transmission privacy in the FL pipeline.

## 2. Check generated files

The demo creates these files:

```text
results/privacy_comparison_demo.csv
results/privacy_comparison_demo.json
results/privacy_payload_preview.json
results/privacy_payload_comparison.png
results/privacy_raw_vs_weights.png
```

Open this CSV:

```powershell
notepad results\privacy_comparison_demo.csv
```

The important rows are:

```text
Centralized Raw Data -> privacy_risk=True
Federated Model Update -> privacy_risk=False
```

## 3. Run the full project with Wireshark packets

If you want `main.py` itself to generate both privacy-proof packets and then continue into the normal training pipeline, run:

```powershell
python main.py
```

This now runs in this order:

```text
1. Centralized raw-data socket demo on port 5001
2. Federated model-update socket demo on port 5000
3. Main centralized training
4. Main federated training on port 5000
5. Final comparison charts
```

If you only want the original training pipeline without the extra privacy demo:

```powershell
python main.py --mode both
```

Important:

- Port `5001` is only used by the centralized raw-data demo.
- Port `5000` is used by the federated model-update demo and by the main federated training.
- The centralized training stage itself does not use sockets because it trains on data already loaded on the same machine.

## 4. Open the dashboard

```powershell
start review_dashboard.html
```

Use these sections during the review:

- Privacy Verification Visual
- Centralized vs Federated Payload Demo
- Packet-Level Privacy Evidence
- Performance vs Privacy

## 5. Optional Wireshark live demo

Open Wireshark and choose:

```text
Adapter for loopback traffic capture
```

For centralized raw-data demo, use this filter:

```text
tcp.port == 5001
```

Then run:

```powershell
python main.py --mode privacy_demo
```

Click a packet with `Len > 0`, then:

```text
Right click -> Follow -> TCP Stream
```

At the bottom of the TCP stream window, set:

```text
Show as: ASCII
```

You should be able to read fields such as:

```text
raw_sensor_data
activity_labels
subject_ids
```

For federated model-update demo, use this filter:

```text
tcp.port == 5000
```

Then run:

```powershell
python main.py --mode privacy_demo
```

Open `Follow TCP Stream` again. This stream is binary pickle data, so it will look like unreadable bytes. That is expected. The decoded terminal output and CSV prove the object keys are:

```text
client_id
weights
num_samples
description
```

What Wireshark proves:

- traffic actually crosses the local socket
- centralized demo uses port 5001
- federated demo uses port 5000

What the CSV proves:

- centralized payload contains raw sensor and label fields
- federated payload contains only model-update fields

## 6. Final line to say to the panel

```text
The centralized demo shows that raw wearable sensor windows and labels are exposed to the server. In the federated demo, the transmitted payload contains only client_id, weights, and num_samples. Therefore, raw HAR sensor data remains local to the client, while the global model still achieves near-centralized accuracy.
```
