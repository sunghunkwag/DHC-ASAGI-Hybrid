"""
Basic usage example of the DHC-ASAGI Hybrid System.
Demonstrates the four integration modes.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

from hybrid import HybridMetaController, HybridConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dummy_data(num_samples=100, batch_size=16):
    """Create dummy dataset for demonstration."""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def demonstrate_modes():
    """Demonstrate all four integration modes."""
    
    logger.info("="*60)
    logger.info("DHC-ASAGI Hybrid System - Basic Usage Example")
    logger.info("="*60)
    
    config = HybridConfig()
    
    logger.info("\nInitializing Hybrid Meta-Controller...")
    hybrid_system = HybridMetaController(config)
    
    logger.info(f"Total parameters: {hybrid_system.num_parameters:,}")
    logger.info(f"DHC-SSM parameters: {hybrid_system.dhc_ssm.num_parameters:,}")
    logger.info(f"ASAGI parameters: {hybrid_system.asagi.num_parameters:,}")
    
    dataloader = create_dummy_data()
    batch = next(iter(dataloader))
    images, labels = batch
    
    modes = ["independent", "feature_sharing", "goal_driven", "deep_integration"]
    
    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"{'='*60}")
        
        hybrid_system.set_mode(mode)
        
        with torch.no_grad():
            outputs = hybrid_system(images, mode=mode)
        
        logger.info(f"Output keys: {list(outputs.keys())}")
        
        if 'logits' in outputs:
            logger.info(f"Logits shape: {outputs['logits'].shape}")
            preds = outputs['logits'].argmax(dim=1)
            accuracy = (preds == labels).float().mean()
            logger.info(f"Batch accuracy: {accuracy:.4f}")
        
        if 'asagi_output' in outputs:
            asagi_out = outputs['asagi_output']
            logger.info(f"Intrinsic signals: {list(asagi_out['intrinsic_signals'].keys())}")
            logger.info(f"Goals shape: {asagi_out['goals'].shape}")
        
        if 'causal_output' in outputs and outputs['causal_output']:
            causal_out = outputs['causal_output']
            logger.info(f"Causal graph shape: {causal_out['causal_graph'].shape}")
            logger.info(f"Graph sparsity: {causal_out['graph_sparsity']:.4f}")


def main():
    """Main function."""
    demonstrate_modes()
    logger.info("\n" + "="*60)
    logger.info("Basic usage demonstration complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
