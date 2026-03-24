Stage 6 — Automation and CI/CD
Overview
Stage 6 transforms the pipeline into a production-ready system by introducing containerisation, automated validation, and code quality enforcement. The goal is a deployment that runs consistently across environments and can be maintained without manual intervention.

Docker Containerisation
The application is packaged into a Docker container that bundles the FastAPI service, the trained model, and all dependencies into a single portable unit.
Model Packaging
Rather than relying on a live MLflow tracking server at runtime, the model is packaged locally within the repository at:
04-deployment/packaged_model/
This decouples the serving environment from the experiment tracking environment, making the container self-contained and reproducible.
Building and Running
bashdocker build -f 06-cicd/Dockerfile -t inventory-api .
docker run -p 8000:8000 inventory-api

CI/CD with GitHub Actions
A GitHub Actions workflow runs automatically on every push. It handles:

Dependency installation — ensures the environment is reproducible from requirements.txt
Linting — runs flake8 to enforce code style across the project

The workflow configuration lives in .github/workflows/ and is triggered on every push to the main branch.

Code Quality
flake8 is configured project-wide via .flake8 and enforced in CI. This ensures consistent style across all pipeline stages, not just the deployment layer.