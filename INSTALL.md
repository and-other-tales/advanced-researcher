# Advanced Researcher Installation Guide

This guide provides instructions for setting up the Advanced Researcher project with the updated dependencies.

## Prerequisites

- Python 3.10+ (recommended: Python 3.11)
- Node.js 18+ (recommended: Node.js 20)
- Poetry (for Python dependency management)
- Yarn or npm (for frontend dependency management)

## Backend Setup

### Using Poetry (Recommended)

1. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

2. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Using pip

If you prefer using pip directly:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   yarn install
   # OR
   npm install
   ```

3. Build the frontend:
   ```bash
   yarn build
   # OR
   npm run build
   ```

## Running the Application

### Running Locally

1. Start the backend:
   ```bash
   # From the project root
   ./run_local.sh
   # OR
   python main.py
   ```

2. In a separate terminal, start the frontend development server:
   ```bash
   cd frontend
   yarn dev
   # OR
   npm run dev
   ```

3. Access the application at http://localhost:3000

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t advanced-researcher .
   ```

2. Run the container:
   ```bash
   docker run -p 8080:8080 advanced-researcher
   ```

3. Access the application at http://localhost:8080

## Security Considerations

This version includes critical security updates to address:

1. Next.js authentication bypass vulnerability (CVE-2025-29927)
2. LangChain remote code execution vulnerabilities
3. Other dependency vulnerabilities

The security improvements include:
- Updated all dependencies to latest secure versions
- Added security headers to prevent common attacks
- Implemented middleware to block malicious requests

## Troubleshooting

### Dependency Conflicts

If you encounter dependency conflicts with LangChain packages:

1. Clean your virtual environment
2. Ensure you have the exact versions specified in requirements.txt
3. Install packages in the correct order:
   ```bash
   pip install langchain-core==0.3.59
   pip install -r requirements.txt
   ```

### Next.js Build Issues

If you encounter build issues with Next.js:

1. Clear the Next.js cache:
   ```bash
   cd frontend
   rm -rf .next
   yarn dev
   ```

### Weaviate Client Migration

If you encounter issues with the updated Weaviate client (v4.x):

1. Review the migration guide at https://weaviate.io/developers/weaviate/client-libraries/python/v3_v4_migration
2. Update your code according to the v4 API changes