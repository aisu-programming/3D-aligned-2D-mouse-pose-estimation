import sleap

labels = sleap.Labels.load_coco("datasets/MARS/MARS_front_COCO.json", "datasets/MARS")
sleap.Labels.save_file(labels, "MARS_front.slp")