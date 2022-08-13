
Learner events
==============

## Network  
  - Valid Packet Received
    - Is Worker Initialisation
      - Send initial noise vectors and signal a worker model download
    - Is Episode Return
      - Compute Gradient Partial and Signal partial gradient received
      - The computed gradient partial may be for an older model and that will need to be compensated for (bother Aech), the Partial Gradient buffer should always be relevant to the current model
    - Is Unknown Packet
      - Do nothing (log)
  - Invalid Packet Received
    - Do nothing (log)
  - Worker Connected
    - Queue a timed signal "Worker initialisation timeout"
  - Worker Disconnected
    - Remove any active downloads for this worker

## Signals
  - Signal: Worker Initialisation Timeout
    - If the worker has not requested initialisation, disconnect the worker
    - Otherwise, do nothing
  - Signal: Partial Gradient Received
    - Add partial gradient to buffer
      - Is buffer full 
        - If the gradient update is still being computed in the background then drop this partial and log a warning, this situation means we've filled two buffers and the gradient update is taking longer than expected.
        - Else, Move the buffer to the gradient update in the background so that new partials can be received while the buffer is being processed
          - Clear the buffer / Create a new buffer
          - Begin computing new gradient update in background
            - Update model and signal new model
  - Signal: Worker Model Download
    - There is an active worker model download of a recent version
      - Do nothing (log)
      - This counts as a skipped / dropped model update to that worker, the worker will now be producing returns with an old policy
    - There is an active worker model download of a very old version
      - Do nothing, the worker model download will be reset on the next Worker Model Download Chunk Signal
    - There is no active worker model download 
      - Add a new transfer to the active model download list 
      - Queue a Worker Model Download Chunk Signal 
  - Signal: Worker Model Download Chunk
    - This model is now very old
      - Reset the model download and send the latest model
    - This model is recent
      - Send the model chunk
      - The model download is complete
        - Drop the model download from the active model download list 
      - The model download is not complete 
        - Signal Worker Model Download Chunk
  - Signal: Model Update 
    - For each worker, Signal Worker Model Download 