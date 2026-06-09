## Conversion Workflow

1. Perform distributed inference on calibration datasets, capturing per-layer activations and final-layer attention scores.

   ```python
   bash forward_calibration.sh
   ```
2. Parsing distributed inference results and computing TokenScore.

   ```python
   python parse_forward_out.py
   ```
3. Compute neuron domain score vectors layer-by-layer based on domain samples and the TokenScore-weighted importance formula.

   ```python
   python calculate_neuron_scores.py
   ```
4. Layer-wise neuron clustering (using domain score vectors as distance) to assign cluster_ids (a list of expert indices per layer). For each split expert, domain coverage is calculated: a domain dimension scores 1 if the average neuron score in that sub-domain exceeds the top 50%, and 0 otherwise.

   ```python
   python split_expert_mlp.py
   ```
5. Converting Qwen3 MLP to Den2MoEESparseMoeBlock layer-by-layer: initialization of shared/routed/zero-experts and router parameters.

   ```python
   python convert_den2moee.py
   ```


