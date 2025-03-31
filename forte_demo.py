import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import logging
from forte_api import ForteOODDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ForteDemo")

def save_dataset_as_png(dataset, save_dir, num_images=1000):
    """
    Save a subset of a dataset as PNG images.
    
    Args:
        dataset: PyTorch dataset
        save_dir (str): Directory to save images
        num_images (int): Number of images to save
    
    Returns:
        list: List of paths to saved images
    """
    logger.info(f"Saving {min(num_images, len(dataset))} images to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    
    for i in tqdm(range(min(num_images, len(dataset))), desc=f"Saving images to {save_dir}"):
        image, label = dataset[i]
        # Convert tensor to PIL Image
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Save the image
        path = os.path.join(save_dir, f"{i}_label{label}.png")
        image.save(path)
        paths.append(path)
    
    return paths

def load_cifar_datasets():
    """
    Load CIFAR10 and CIFAR100 datasets.
    
    Returns:
        tuple: CIFAR10 train and test sets, CIFAR100 test set
    """
    logger.info("Loading CIFAR10 and CIFAR100 datasets...")
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load CIFAR10 train and test sets
    cifar10_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    cifar10_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Load CIFAR100 test set
    cifar100_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    
    logger.info(f"Loaded datasets - CIFAR10 train: {len(cifar10_train)} images, " +
               f"CIFAR10 test: {len(cifar10_test)} images, " +
               f"CIFAR100 test: {len(cifar100_test)} images")
    
    return cifar10_train, cifar10_test, cifar100_test

def print_training_phases():
    """Print information about the phases of the Forte training pipeline."""
    phases = [
        ("1. Data Preparation", 
         "Convert datasets to image files and prepare directories"),
        
        ("2. Feature Extraction", 
         "Extract semantic features using pretrained models (CLIP, ViTMSN, DINOv2)"),
        
        ("3. PRDC Computation", 
         "Compute Precision, Recall, Density, Coverage metrics from extracted features"),
        
        ("4. Detector Training", 
         "Train OOD detector (GMM, KDE, or OCSVM) on PRDC features"),
        
        ("5. Evaluation", 
         "Compute scores and performance metrics on test datasets")
    ]
    
    logger.info("\n=== Forte OOD Detection Pipeline ===")
    for i, (phase, desc) in enumerate(phases):
        logger.info(f"{phase}: {desc}")
    logger.info("="*40)

def main(args):
    # Print pipeline phases information
    print_training_phases()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    logger.info(f"Running with configuration: {args}")
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs(args.embedding_dir, exist_ok=True)
    
    # Phase 1: Data Preparation
    logger.info("\n=== Phase 1: Data Preparation ===")
    cifar10_train, cifar10_test, cifar100_test = load_cifar_datasets()
    
    # Create directories for images
    os.makedirs("data/cifar10/train", exist_ok=True)
    os.makedirs("data/cifar10/test", exist_ok=True)
    os.makedirs("data/cifar100/test", exist_ok=True)
    
    # Check if we need to save images
    if not os.path.exists("data/cifar10/train/0_label0.png") or args.force_save:
        logger.info("Converting datasets to PNG images...")
        # Save CIFAR10 training images
        cifar10_train_paths = save_dataset_as_png(
            cifar10_train, "data/cifar10/train", num_images=args.num_train_images
        )
        
        # Save CIFAR10 test images
        cifar10_test_paths = save_dataset_as_png(
            cifar10_test, "data/cifar10/test", num_images=args.num_test_images
        )
        
        # Save CIFAR100 test images
        cifar100_test_paths = save_dataset_as_png(
            cifar100_test, "data/cifar100/test", num_images=args.num_test_images
        )
    else:
        logger.info("Using previously saved images...")
        cifar10_train_paths = sorted([os.path.join("data/cifar10/train", f) 
                                    for f in os.listdir("data/cifar10/train") 
                                    if f.endswith(".png")])[:args.num_train_images]
        
        cifar10_test_paths = sorted([os.path.join("data/cifar10/test", f) 
                                   for f in os.listdir("data/cifar10/test") 
                                   if f.endswith(".png")])[:args.num_test_images]
        
        cifar100_test_paths = sorted([os.path.join("data/cifar100/test", f) 
                                    for f in os.listdir("data/cifar100/test") 
                                    if f.endswith(".png")])[:args.num_test_images]
    
    logger.info(f"Number of CIFAR10 training images: {len(cifar10_train_paths)}")
    logger.info(f"Number of CIFAR10 test images: {len(cifar10_test_paths)}")
    logger.info(f"Number of CIFAR100 test images: {len(cifar100_test_paths)}")
    
    # Phase 2-4: Feature Extraction, PRDC Computation, and Detector Training
    logger.info("\n=== Phase 2-4: Feature Extraction, PRDC Computation, and Detector Training ===")
    start_time = time.time()
    logger.info(f"Creating ForteOODDetector with method: {args.method}, nearest_k: {args.nearest_k}")
    detector = ForteOODDetector(
        batch_size=args.batch_size,
        device=args.device,
        embedding_dir=args.embedding_dir,
        method=args.method,
        nearest_k=args.nearest_k
    )
    
    # Fit the detector - this performs feature extraction, PRDC computation, and detector training
    logger.info(f"Fitting ForteOODDetector on {len(cifar10_train_paths)} in-distribution images...")
    detector.fit(cifar10_train_paths, val_split=0.2, random_state=args.seed)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Phase 5: Evaluation
    logger.info("\n=== Phase 5: Evaluation ===")
    
    # Benchmark on ID data (CIFAR10 test)
    logger.info("Benchmarking detector on CIFAR10 (in-distribution)...")
    start_time = time.time()
    id_scores = detector._get_ood_scores(cifar10_test_paths, cache_name="id_benchmark")
    id_prediction_time = time.time() - start_time
    logger.info(f"ID prediction time for {len(cifar10_test_paths)} images: {id_prediction_time:.2f} seconds " + 
          f"({id_prediction_time/len(cifar10_test_paths):.4f} sec/image)")
    
    # Benchmark on OOD data (CIFAR100 test)
    logger.info("Benchmarking detector on CIFAR100 (out-of-distribution)...")
    start_time = time.time()
    ood_scores = detector._get_ood_scores(cifar100_test_paths, cache_name="ood_benchmark")
    ood_prediction_time = time.time() - start_time
    logger.info(f"OOD prediction time for {len(cifar100_test_paths)} images: {ood_prediction_time:.2f} seconds " + 
          f"({ood_prediction_time/len(cifar100_test_paths):.4f} sec/image)")
    
    # Score statistics
    logger.info("\nScore Statistics:")
    logger.info(f"CIFAR10 (ID)  - Mean: {np.mean(id_scores):.4f}, Std: {np.std(id_scores):.4f}, " + 
          f"Min: {np.min(id_scores):.4f}, Max: {np.max(id_scores):.4f}")
    logger.info(f"CIFAR100 (OOD) - Mean: {np.mean(ood_scores):.4f}, Std: {np.std(ood_scores):.4f}, " + 
          f"Min: {np.min(ood_scores):.4f}, Max: {np.max(ood_scores):.4f}")
    
    # Calculate threshold based on ID scores
    threshold = np.percentile(id_scores, 5)  # 5th percentile
    logger.info(f"Suggested decision threshold (5th percentile of ID scores): {threshold:.4f}")
    
    # Calculate detection accuracy
    id_correct = (id_scores > threshold).mean()
    ood_correct = (ood_scores <= threshold).mean() 
    overall_acc = (id_correct * len(id_scores) + ood_correct * len(ood_scores)) / (len(id_scores) + len(ood_scores))
    logger.info(f"ID Detection Rate: {id_correct:.4f}, OOD Detection Rate: {ood_correct:.4f}")
    logger.info(f"Overall Accuracy: {overall_acc:.4f}")
    
    # Full evaluation on mixed test set
    logger.info("\nPerforming full evaluation on CIFAR10/CIFAR100 test sets...")
    evaluation_start_time = time.time()
    results = detector.evaluate(cifar10_test_paths, cifar100_test_paths)
    evaluation_time = time.time() - evaluation_start_time
    
    # Print performance metrics
    logger.info("\n=== OOD Detection Performance ===")
    logger.info(f"Method: {args.method}, Nearest_k: {args.nearest_k}")
    logger.info(f"AUROC: {results['AUROC']:.4f}")
    logger.info(f"FPR@95TPR: {results['FPR@95TPR']:.4f}")
    logger.info(f"AUPRC: {results['AUPRC']:.4f}")
    logger.info(f"F1 Score: {results['F1']:.4f}")
    logger.info(f"Evaluation time: {evaluation_time:.2f} seconds")
    
    # Visualize results
    if args.visualize:
        logger.info("\nGenerating visualizations...")
        
        # Plot score distributions
        plt.figure(figsize=(10, 6))
        bins = np.linspace(min(np.min(id_scores), np.min(ood_scores)), 
                           max(np.max(id_scores), np.max(ood_scores)), 
                           30)
        
        plt.hist(id_scores, bins=bins, alpha=0.7, label='CIFAR10 (In-Distribution)', density=True)
        plt.hist(ood_scores, bins=bins, alpha=0.7, label='CIFAR100 (Out-of-Distribution)', density=True)
        
        # Add threshold line
        plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.4f})')
        
        plt.legend()
        plt.title(f'ForteOODDetector Scores ({args.method}, nearest_k={args.nearest_k})')
        plt.xlabel('OOD Score (higher = more in-distribution like)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(f"forte_{args.method}_results.png")
        logger.info(f"Score distribution saved to forte_{args.method}_results.png")
        
        # Show examples with predictions
        num_examples = min(5, len(cifar10_test_paths), len(cifar100_test_paths))
        
        fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
        
        # CIFAR10 examples (should be classified as in-distribution)
        for i in range(num_examples):
            img = Image.open(cifar10_test_paths[i])
            axes[0, i].imshow(img)
            
            score = id_scores[i]
            is_id = score > threshold
            correct = is_id  # For ID samples, prediction is correct if classified as ID
            
            color = 'green' if correct else 'red'
            pred = "ID" if is_id else "OOD"
            axes[0, i].set_title(f"CIFAR10 (true=ID)\nPred: {pred}\nScore: {score:.2f}", color=color)
            axes[0, i].axis('off')
        
        # CIFAR100 examples (should be classified as out-of-distribution)
        for i in range(num_examples):
            img = Image.open(cifar100_test_paths[i])
            axes[1, i].imshow(img)
            
            score = ood_scores[i]
            is_id = score > threshold
            correct = not is_id  # For OOD samples, prediction is correct if classified as OOD
            
            color = 'green' if correct else 'red'
            pred = "ID" if is_id else "OOD"
            axes[1, i].set_title(f"CIFAR100 (true=OOD)\nPred: {pred}\nScore: {score:.2f}", color=color)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig("forte_examples.png")
        logger.info("Example predictions saved to forte_examples.png")
        
        # ROC curve
        plt.figure(figsize=(8, 6))
        
        # Create labels (1 for ID, 0 for OOD)
        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        scores_combined = np.concatenate([id_scores, ood_scores])
        
        # Calculate ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(labels, scores_combined)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # Mark the FPR at 95% TPR
        idx_95tpr = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95tpr = fpr[idx_95tpr]
        plt.scatter(fpr_at_95tpr, 0.95, color='red', 
                   label=f'FPR@95TPR = {fpr_at_95tpr:.4f}', zorder=5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {args.method.upper()}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.savefig(f"forte_{args.method}_roc.png")
        logger.info(f"ROC curve saved to forte_{args.method}_roc.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forte OOD Detection Demo")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    parser.add_argument("--method", type=str, default="gmm", choices=["gmm", "kde", "ocsvm"], 
                        help="OOD detection method")
    parser.add_argument("--nearest_k", type=int, default=5, help="Number of nearest neighbors for PRDC")
    parser.add_argument("--num_train_images", type=int, default=1000, help="Number of training images")
    parser.add_argument("--num_test_images", type=int, default=500, help="Number of test images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--force_save", action="store_true", help="Force save images even if they exist")
    parser.add_argument("--embedding_dir", type=str, default="embeddings", help="Directory to store embeddings")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.getLogger().setLevel(numeric_level)
    
    main(args)