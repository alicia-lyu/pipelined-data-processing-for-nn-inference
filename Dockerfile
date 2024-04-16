# Use NVIDIA Triton Server image
FROM nvcr.io/nvidia/tritonserver:24.03-py3

# Set the working directory
WORKDIR /models

# Copy the model repository from the local directory into the container
# Note: Ensure that the model repository is in the context directory of the Docker build or adjust the path accordingly.
COPY ./triton-tutorial/model_repository /models

# Set the shared memory size via environment variable (may need to be handled at runtime instead, depending on the deployment environment)
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set the command to run the Triton Server
CMD ["tritonserver", "--model-repository=/models", "--model-control-mode=poll"]