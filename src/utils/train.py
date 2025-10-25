import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from src.data.data_prep import prepare_data
from src.utils.preprocess import get_preprocessor, get_dataloader
from src.models.segmenter import CellSegmenter
from src.config import trainConfig, gpuConfig, DataPaths

def training_pipeline(config=trainConfig()):
    train_img, val_img, train_mask, val_mask = prepare_data(config)
    preprocess = get_preprocessor(config.model_name)
    
    train_loader = get_dataloader(train_img, train_mask, preprocess, config.batch_size)
    val_loader = get_dataloader(val_img, val_mask, preprocess, config.batch_size, shuffle=False)
    
    model = CellSegmenter(config.model_name, config.num_labels)
    if config.num_gpus > 1:
        model = DataParallel(model)
    model = model.to(gpuConfig.device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(trainConfig.num_epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(config.device), masks.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{config.num_epochs} completed.")
    
    torch.save(model.state_dict(), "model.pth")
    return model