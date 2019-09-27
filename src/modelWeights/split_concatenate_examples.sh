# split -b 50M -d retinanet_pins_inference.h5 "retinanet_pins_inference.h5.part_"
# cat retinanet_pins_inference.h5.part_* > retinanet_pins_inference_restored.h5
# cmp retinanet_pins_inference.h5 retinanet_pins_inference_restored.h5