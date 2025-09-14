#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---

# GCP Project ID
GCP_PROJECT_ID=""
# GCP Region
GCP_REGION="us-central1"
# Artifact Registry repository name
AR_REPO_NAME="ai-caption-generator"
# Cloud Build machine type
MACHINE_TYPE="E2_HIGHCPU_8"

# --- Helper Functions ---

# Function to print messages
function print_message() {
    echo "🔵 $1"
}

# Function to print errors
function print_error() {
    echo "🔴 ERROR: $1" >&2
    exit 1
}

# --- Pre-flight Checks ---

print_message "Checking for gcloud CLI..."
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install it from https://cloud.google.com/sdk/docs/install"
fi

print_message "Checking gcloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_message "You are not logged in to gcloud. Running 'gcloud auth login'..."
    gcloud auth login
    gcloud auth application-default login
fi

# --- User Input ---

if [ -z "$GCP_PROJECT_ID" ]; then
    GCP_PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$GCP_PROJECT_ID" ]; then
        read -p "Enter your GCP Project ID: " GCP_PROJECT_ID
        if [ -z "$GCP_PROJECT_ID" ]; then
            print_error "GCP Project ID is required."
        fi
    else
        print_message "Using existing gcloud project: $GCP_PROJECT_ID"
        read -p "Do you want to use this project? (y/N): " confirm
        if [[ ! "$confirm" =~ ^[yY]([eE][sS])?$ ]]; then
            read -p "Enter your GCP Project ID: " GCP_PROJECT_ID
        fi
    fi
fi

gcloud config set project "$GCP_PROJECT_ID"
print_message "Set GCP project to $GCP_PROJECT_ID"

read -p "Enter the GCP region to deploy to (default: $GCP_REGION): " user_region
if [ -n "$user_region" ]; then
    GCP_REGION=$user_region
fi
gcloud config set run/region "$GCP_REGION"
gcloud config set compute/region "$GCP_REGION"
print_message "Set GCP region to $GCP_REGION"


# --- Enable APIs ---

print_message "Enabling necessary GCP APIs..."
APIS_TO_ENABLE=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "iam.googleapis.com"
)

for api in "${APIS_TO_ENABLE[@]}"; do
    if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q .; then
        print_message "Enabling $api..."
        gcloud services enable "$api"
    else
        print_message "$api is already enabled."
    fi
done

# --- Artifact Registry ---

print_message "Checking for Artifact Registry repository..."
if ! gcloud artifacts repositories describe "$AR_REPO_NAME" --location="$GCP_REGION" --project="$GCP_PROJECT_ID" &> /dev/null; then
    print_message "Creating Artifact Registry repository '$AR_REPO_NAME' in '$GCP_REGION'..."
    gcloud artifacts repositories create "$AR_REPO_NAME" \
        --repository-format=docker \
        --location="$GCP_REGION" \
        --description="Docker repository for AI Image Caption Generator"
else
    print_message "Artifact Registry repository '$AR_REPO_NAME' already exists."
fi


# --- Cloud Build ---

print_message "Starting Cloud Build process..."
gcloud builds submit --config cloudbuild.yaml \
    --substitutions=_REGION="$GCP_REGION",_AR_REPO_NAME="$AR_REPO_NAME" \
    --machine-type="$MACHINE_TYPE"

# --- Post-Deployment ---

print_message "Deployment complete!"
FRONTEND_URL=$(gcloud run services describe ai-caption-frontend --platform managed --region "$GCP_REGION" --format 'value(status.url)')
BACKEND_URL=$(gcloud run services describe ai-caption-backend --platform managed --region "$GCP_REGION" --format 'value(status.url)')

echo ""
echo "✅ Frontend service URL: $FRONTEND_URL"
echo "✅ Backend service URL: $BACKEND_URL"
echo ""
print_message "It may take a few minutes for the services to be fully available."
