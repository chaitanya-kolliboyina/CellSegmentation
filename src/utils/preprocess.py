from transformers import SegformerImageProcessor

def get_preprocessor(model_name):
    processor = SegformerImageProcessor.from_pretrained(model_name)
    
    def preprocess_pipeline(image, mask=None):
        if mask is not None:
            inputs = processor(images=image, segmentation_maps=mask, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), inputs["labels"].squeeze(0)
        else:
            inputs = processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), None
    
    return preprocess_pipeline

def get_dataloader(img_paths, mask_paths, preprocess_pipeline, batch_size, shuffle=True):
    dataset = SegmentationDataset(img_paths, mask_paths, transform=preprocess_pipeline)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)