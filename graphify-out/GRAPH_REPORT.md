# Graph Report - transformer-caption-ptbr  (2026-10-05)

## Corpus Check
- 83 files · ~303,771 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 18 file(s) not represented in the graph (top: (none) 9, .css 5, .template 3)

## Summary
- 617 nodes · 994 edges · 61 communities (27 shown, 34 thin omitted)
- Extraction: 93% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Core Pipeline Code
- Flickr30K-PTBR Dataset & Cards
- VLM Train/Eval Entrypoints
- VLM Data Collators
- VED Config & Requirements
- Zero-Shot Metrics Utils
- Bulma Carousel JS
- Metrics Analysis Eval
- LLM-Judge Metrics
- CLIP Score Metrics
- VED Metrics Utils
- Analyzer & HF Inference
- VLM Data Processing
- Model-as-Evaluator Eval
- Zero-Shot Data Processing
- VLM LoRA Config
- VLM Evaluation Metrics
- VLM Eval Script
- Judge Eval Script
- Encoder-Decoder Architecture Figure
- Zero-Shot Inference
- Zero-Shot Eval Predictions
- Transformers Inference
- Generation Collators
- Conversation Templates
- Transformers Blueprint Illustration
- Judge Config & Datasets
- Zero-Shot LoRA Config
- VLM Metrics Module
- Metrics Analysis Config
- Bulma Slider JS
- VLM Overview Graphics
- Carousel Minified JS
- VED Model Config
- Zero-Shot Model Config
- Swin-DistilBERT Qualitative Results
- Swin-GPT-2 Qualitative Results
- Slider Minified JS
- CLIP Score Helper
- Related Work References
- Minor: ved/setup.sh
- Minor: vlm/setup.sh
- Minor: setup-aac.sh
- Minor: vlm_zero_shot/setup.sh

## God Nodes (most connected - your core abstractions)
1. `_classCallCheck()` - 14 edges
2. `qc()` - 13 edges
3. `Transformer-Based Vision Models for Brazilian Portuguese Image Captioning (project)` - 12 edges
4. `train_model()` - 11 edges
5. `VED models for Brazilian Portuguese captioning` - 10 edges
6. `ul()` - 9 edges
7. `Swin-DistilBERTimbau model card` - 9 edges
8. `bc()` - 8 edges
9. `pl()` - 8 edges
10. `generate_grouped_dataset()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Encoder-decoder architecture diagram` --semantically_similar_to--> `Example elephant captioning image`  [AMBIGUOUS] [semantically similar]
  docs/static/images/architecture.png → cards/models/example.png
- `VLM zero-shot module README (VED-template content)` --semantically_similar_to--> `VED models for Brazilian Portuguese captioning`  [INFERRED] [semantically similar]
  vlm_zero_shot/README.md → ved/README.md
- `model_as_evaluator config.yaml` --semantically_similar_to--> `metrics_analysis config.yaml`  [INFERRED] [semantically similar]
  model_as_evaluator/config.yaml → metrics_analysis/config.yaml
- `vlm requirements.txt` --semantically_similar_to--> `ved requirements.txt`  [INFERRED] [semantically similar]
  vlm/requirements.txt → ved/requirements.txt
- `vlm_zero_shot requirements.txt` --semantically_similar_to--> `vlm requirements.txt`  [INFERRED] [semantically similar]
  vlm_zero_shot/requirements.txt → vlm/requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Top Swin-based VED captioning models trained on Flickr30K PT** — cards_models_swin_distilbertimbau_model, cards_models_swin_gportuguese_2_model, cards_datasets_flickr30k_pt_br_dataset [EXTRACTED 1.00]
- **VLMs fine-tuned via QLoRA config** — vlm_readme_llama3_vision, vlm_readme_phi3_vision, vlm_readme_paligemma, vlm_readme_qlora [EXTRACTED 1.00]
- **VLMs finetuned with QLoRA config** — vlm_src_config_config_finetuning_llama3_vision, vlm_src_config_config_finetuning_phi3_vision, vlm_src_config_config_finetuning_paligemma, vlm_src_config_config_finetuning_qlora [EXTRACTED 0.95]
- **VLM finetune/inference/evaluation config trio** — vlm_src_config_config_finetuning, vlm_src_config_config_inference, vlm_src_config_config_evaluation [INFERRED 0.75]
- **Vision encoder-decoder captioning pipeline** — docs_static_images_architecture_visual_encoder, docs_static_images_architecture_cross_attention, docs_static_images_architecture_language_decoder [EXTRACTED 1.00]
- **Portuguese captioning related work** — docs_static_images_references_pracegover, docs_static_images_references_translated_dataset, docs_static_images_references_grit [INFERRED 0.85]
- **VLMs evaluated for Brazilian Portuguese captioning** — images_models_llama_3_2_vision, images_models_phi_3_vision, images_models_paligemma [INFERRED 0.85]

## Communities (61 total, 34 thin omitted)

### Community 0 - "Core Pipeline Code"
Cohesion: 0.10
Nodes (47): a(), Ac(), Al(), bc(), bl(), c(), cl(), dc() (+39 more)

### Community 1 - "Flickr30K-PTBR Dataset & Cards"
Cohesion: 0.06
Nodes (48): Flickr30K Portuguese Translated (laicsiifes/flickr30k-pt-br), Google Translator API, Karpathy and Fei-Fei 2015 (Deep Visual-Semantic Alignments), Karpathy splits, DistilBERTimbau decoder, Captioning metrics (CIDEr-D, BLEU@4, ROUGE-L, METEOR, BERTScore), Swin-DistilBERTimbau model card, Lower-performing VED variants (DeiT/ViT/BERTimbau/GPorTuguese-2 combinations) (+40 more)

### Community 2 - "VLM Train/Eval Entrypoints"
Cohesion: 0.06
Nodes (19): evaluate_from_model(), evaluate_from_predictions(), train_model(), collate_fn(), get_data_loader(), load_datasets(), preprocess(), transform_datasets() (+11 more)

### Community 3 - "VLM Data Collators"
Cohesion: 0.09
Nodes (3): DataCollatorForGeneration, DataCollatorForTraining, VLMsDataCollator

### Community 4 - "VED Config & Requirements"
Cohesion: 0.09
Nodes (26): ved requirements.txt, Caption metric libs (rouge_score, bert_score, aac-metrics, open_clip), peft and bitsandbytes (QLoRA libs), transformers 4.45.2, ved/src config.yml, Portuguese decoders (BERTimbau, DistilBERT, GPT-2, RoBERTa, BART), Vision encoders (ViT, Swin, BEiT, DeiT), flickr30k-pt-br-human-generated dataset (+18 more)

### Community 5 - "Zero-Shot Metrics Utils"
Cohesion: 0.15
Nodes (4): evaluate_from_predictions(), batch_generation(), batch_generation_from_API(), evaluate_from_predictions()

### Community 6 - "Bulma Carousel JS"
Cohesion: 0.16
Nodes (16): Autoplay(), Breakpoints(), bulmaCarousel(), _classCallCheck(), Coordinate(), _defineProperty(), EventEmitter(), Fade() (+8 more)

### Community 7 - "Metrics Analysis Eval"
Cohesion: 0.16
Nodes (13): compute_all_metrics(), compute_individual_metric(), compute_individual_metrics(), compute_metrics_sample(), map_item(), evaluate_captions(), compute_bert_scores(), compute_bleu_scores() (+5 more)

### Community 8 - "LLM-Judge Metrics"
Cohesion: 0.14
Nodes (4): clip_score(), compute_clip_scores(), compute_vlm_as_a_judge(), ref_clip_score()

### Community 9 - "CLIP Score Metrics"
Cohesion: 0.15
Nodes (4): compute_clip_scores(), clip_score(), compute_clip_scores(), ref_clip_score()

### Community 10 - "VED Metrics Utils"
Cohesion: 0.15
Nodes (4): batch_decode_filter(), compute_metrics(), compute_rouge(), train_model()

### Community 11 - "Analyzer & HF Inference"
Cohesion: 0.22
Nodes (6): analyze(), generate_grouped_dataset(), join_datasets(), load_datasets(), select_correct_data(), select_incorrect_data()

### Community 12 - "VLM Data Processing"
Cohesion: 0.18
Nodes (8): image_to_message(), load_datasets(), preprocess(), preprocess_for_API(), map_item(), map_item(), transform_datasets(), encode_image()

### Community 13 - "Model-as-Evaluator Eval"
Cohesion: 0.23
Nodes (4): compute_all_judges(), compute_individual_metrics(), evaluate_predictions(), compute_llm_as_a_judge()

### Community 14 - "Zero-Shot Data Processing"
Cohesion: 0.18
Nodes (7): image_to_message(), load_datasets(), preprocess(), preprocess_for_API(), map_item(), map_item(), transform_datasets()

### Community 15 - "VLM LoRA Config"
Cohesion: 0.17
Nodes (4): config_vars(), configure_model_and_processor(), create_lora_config(), ViTucanoProcessor

### Community 16 - "VLM Evaluation Metrics"
Cohesion: 0.24
Nodes (4): compute_all_metrics(), compute_individual_metrics(), compute_invidiual_metric(), evaluate_predictions()

### Community 19 - "Encoder-Decoder Architecture Figure"
Cohesion: 0.20
Nodes (11): Caption: Um elefante esta parado em um campo, Example elephant captioning image, Caption Output, Randomly Initialized Cross-Attention Layers, Encoder-decoder architecture diagram, Image Input (Patches), Language Decoder, Pre-Trained Transformer-Based Language Model (+3 more)

### Community 21 - "Zero-Shot Eval Predictions"
Cohesion: 0.27
Nodes (4): compute_all_metrics(), compute_individual_metrics(), compute_invidiual_metric(), evaluate_predictions()

### Community 27 - "Transformers Blueprint Illustration"
Cohesion: 0.22
Nodes (9): Image Captioning with Transformers blueprint, Self-attention and cross-attention, Image features (CNN) input, Deployment pipeline (train, validate, test, inference), Transformer encoder-decoder architecture, Evaluation metrics (BLEU, METEOR, CIDEr, ROUGE-L), Training: loss, Adam optimizer, warmup, hyperparameter studies, Brazilian Transformers robot illustration (+1 more)

### Community 28 - "Judge Config & Datasets"
Cohesion: 0.28
Nodes (9): metrics_analysis config.yaml, Correct/incorrect caption sampling, flickr30k-pt-br-5k-human-generated (native), flickr30k-pt-br-5k (translated), model_as_evaluator config.yaml, CLAIR LLM-as-a-Judge template, flickr30k-pt-br-5k-human-generated (native), flickr30k-pt-br-5k (translated) (+1 more)

### Community 29 - "Zero-Shot LoRA Config"
Cohesion: 0.22
Nodes (3): config_vars(), configure_model_and_processor(), create_lora_config()

### Community 32 - "Bulma Slider JS"
Cohesion: 0.36
Nodes (4): bulmaSlider(), _classCallCheck(), EventEmitter(), _possibleConstructorReturn()

### Community 33 - "VLM Overview Graphics"
Cohesion: 0.38
Nodes (7): Dog image captioning illustration, Image captioning (labeling image as dog), VLM models overview graphic, Brazilian Portuguese image captioning, LLaMa 3.2 Vision, PaliGemma, Phi-3 Vision

### Community 37 - "Swin-DistilBERT Qualitative Results"
Cohesion: 0.40
Nodes (5): Flickr qualitative results figure, Simple descriptions and generic scenes, Human reference captions, Shorter captions (~12 words), lower variance (~5 words), Swin-DistilBERT model

### Community 38 - "Swin-GPT-2 Qualitative Results"
Cohesion: 0.50
Nodes (5): Qualitative results figure (Swin-GPT-2 vs human captions), Dataset caption traits (long captions, high variance, names, world knowledge), Failure modes (text-heavy images, mismatch, linguistic errors), Human reference captions (Portuguese), Swin-GPT-2

### Community 41 - "Related Work References"
Cohesion: 0.50
Nodes (4): Related work references figure, Image captioning for Brazilian Portuguese using GRIT (CoRR'24), #PraCegoVer dataset paper (MDPI Data'22), Towards Image Captioning for Portuguese: Translated Dataset (ICEIS'22)

## Ambiguous Edges - Review These
- `VLM Zero/Few-Shot module` → `VLM zero-shot module README (VED-template content)`  [AMBIGUOUS]
  vlm_zero_shot/README.md · relation: conceptually_related_to
- `Example elephant captioning image` → `Encoder-decoder architecture diagram`  [AMBIGUOUS]
  docs/static/images/architecture.png · relation: semantically_similar_to

## Knowledge Gaps
- **63 isolated node(s):** `setup.sh script`, `setup-aac.sh script`, `setup.sh script`, `setup.sh script`, `LAICSI-IFES` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 286 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `VLM Zero/Few-Shot module` and `VLM zero-shot module README (VED-template content)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `DataCollatorForGeneration` connect `VLM Data Collators` to `Conversation Templates`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **What connects `setup.sh script`, `setup-aac.sh script`, `setup.sh script` to the rest of the system?**
  _63 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Core Pipeline Code` be split into smaller, more focused modules?**
  _Cohesion score 0.10122448979591837 - nodes in this community are weakly interconnected._
- **What is the exact relationship between `Example elephant captioning image` and `Encoder-decoder architecture diagram`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `DataCollatorForTraining` connect `VLM Data Collators` to `Conversation Templates`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Should `Flickr30K-PTBR Dataset & Cards` be split into smaller, more focused modules?**
  _Cohesion score 0.05585106382978723 - nodes in this community are weakly interconnected._