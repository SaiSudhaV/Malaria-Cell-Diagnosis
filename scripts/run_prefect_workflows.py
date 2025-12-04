#!/usr/bin/env python3
"""Script to run Prefect workflows locally"""

import asyncio
import logging
from prefect import serve
from workflows.prefect_flows import (
    preprocessing_flow,
    evaluation_flow,
    batch_prediction_flow,
    end_to_end_pipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_workflows():
    """Run all workflows"""
    
    # Create deployments
    deployments = [
        preprocessing_flow.to_deployment(
            name="data-preprocessing",
            tags=["data", "preprocessing"],
            description="Data loading and preprocessing workflow"
        ),
        evaluation_flow.to_deployment(
            name="model-evaluation", 
            tags=["model", "evaluation"],
            description="Model evaluation workflow"
        ),
        batch_prediction_flow.to_deployment(
            name="batch-prediction",
            tags=["prediction", "batch"],
            description="Batch prediction workflow"
        ),
        end_to_end_pipeline.to_deployment(
            name="end-to-end-pipeline",
            tags=["pipeline", "complete"],
            description="Complete end-to-end ML pipeline"
        )
    ]
    
    # Serve deployments
    await serve(*deployments, limit=10)

def run_single_workflow(workflow_name: str):
    """Run a single workflow"""
    workflows = {
        'preprocessing': preprocessing_flow,
        'evaluation': evaluation_flow,
        'batch_prediction': batch_prediction_flow,
        'end_to_end': end_to_end_pipeline
    }
    
    if workflow_name not in workflows:
        logger.error(f"Unknown workflow: {workflow_name}")
        logger.info(f"Available workflows: {list(workflows.keys())}")
        return
    
    logger.info(f"Running {workflow_name} workflow...")
    result = workflows[workflow_name]()
    logger.info(f"Workflow completed with result: {result}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        workflow_name = sys.argv[1]
        run_single_workflow(workflow_name)
    else:
        logger.info("Starting Prefect server with all deployments...")
        asyncio.run(run_workflows())