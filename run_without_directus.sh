#!/bin/bash

# Clear any Directus-related environment variables
unset DIRECTUS_URL
unset DIRECTUS_TOKEN
unset DIRECTUS_STORAGE_GCS_CREDENTIALS
unset DIRECTUS_DB_PASSWORD
unset DIRECTUS_ADMIN_PASSWORD
unset DIRECTUS_ADMIN_EMAIL
unset DIRECTUS_KEY
unset DIRECTUS_SECRET
unset MIXTURE_DIRECTUS_TOKEN

# Run the application (modify this command based on how you normally start the app)
cd frontend && npm run dev