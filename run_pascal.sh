python process_pascal.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --output_dir "test" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --device "cuda" \
  --img_list $1 \
  --im_path $2 \



