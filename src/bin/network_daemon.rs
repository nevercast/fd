// Learner events
// ==============

// ## Network
//   - Valid Packet Received
//     - Is Worker Initialisation
//       - Send initial noise vectors and signal a worker model download
//     - Is Episode Return
//       - Compute Gradient Partial and Signal partial gradient received
//       - The computed gradient partial may be for an older model and that will need to be compensated for (bother Aech), the Partial Gradient buffer should always be relevant to the current model
//     - Is Unknown Packet
//       - Do nothing (log)
//   - Invalid Packet Received
//     - Do nothing (log)
//   - Worker Connected
//     - Queue a timed signal "Worker initialisation timeout"
//   - Worker Disconnected
//     - Remove any active downloads for this worker

// ## Signals
//   - Signal: Worker Initialisation Timeout
//     - If the worker has not requested initialisation, disconnect the worker
//     - Otherwise, do nothing
//   - Signal: Partial Gradient Received
//     - Add partial gradient to buffer
//       - Is buffer full
//         - If the gradient update is still being computed in the background then drop this partial and log a warning, this situation means we've filled two buffers and the gradient update is taking longer than expected.
//         - Else, Move the buffer to the gradient update in the background so that new partials can be received while the buffer is being processed
//           - Clear the buffer / Create a new buffer
//           - Begin computing new gradient update in background
//             - Update model and signal new model
//   - Signal: Worker Model Download
//     - There is an active worker model download of a recent version
//       - Do nothing (log)
//       - This counts as a skipped / dropped model update to that worker, the worker will now be producing returns with an old policy
//     - There is an active worker model download of a very old version
//       - Do nothing, the worker model download will be reset on the next Worker Model Download Chunk Signal
//     - There is no active worker model download
//       - Add a new transfer to the active model download list
//       - Queue a Worker Model Download Chunk Signal
//   - Signal: Worker Model Download Chunk
//     - This model is now very old
//       - Reset the model download and send the latest model
//     - This model is recent
//       - Send the model chunk
//       - The model download is complete
//         - Drop the model download from the active model download list
//       - The model download is not complete
//         - Signal Worker Model Download Chunk
//   - Signal: Model Update
//     - For each worker, Signal Worker Model Download

use std::sync::Arc;
use std::time::Duration;

use fdlib::common::*;
use fdlib::model::Model;
use fnv::FnvHashMap;
use message_io::network::{Endpoint, NetEvent, Transport};
use message_io::node::{self, NodeEvent};

const WORKER_INITIALISATION_TIMEOUT_MS: u64 = 3000;
const PARAMETER_COUNT: usize = 1_000_000;
const NOISE_BLOCKS: usize = 16;

// MAXIMUM MODEL AGE also applies to dropping received returns from worker episodes.
const MAXIMUM_MODEL_AGE: u32 = 10;

struct ConnectedWorker {
    has_initialised: bool,
}

enum NodeSignal {
    NewConnectedWorker(Endpoint),
    WorkerInitialisationTimeout(Endpoint),
    NextTransferBlock(Endpoint),
    InitialiseWorker(Endpoint),
}

fn main() {
    let mut latest_model_version: ParameterVersion = 0;
    let mut models = FnvHashMap::<ParameterVersion, Arc<Model>>::default();
    let mut active_transfers = FnvHashMap::<Endpoint, ParameterTransfer>::default();
    let mut connected_workers = FnvHashMap::<Endpoint, ConnectedWorker>::default();

    let mut model = Vec::<f32>::with_capacity(PARAMETER_COUNT);
    for i in 0..PARAMETER_COUNT {
        model.push(i as f32);
    }

    models.insert(latest_model_version, Arc::new(Model::new(model)));

    println!("{}", model.len());

    // Create a node, the main message-io entity. It is divided in 2 parts:
    // The 'handler', used to make actions (connect, send messages, signals, stop the node...)
    // The 'listener', used to read events from the network or signals.
    let (handler, listener) = node::split::<NodeSignal>();

    // Listen for TCP, UDP and WebSocket messages at the same time.
    handler
        .network()
        .listen(Transport::FramedTcp, "0.0.0.0:3042")
        .unwrap();
    handler
        .network()
        .listen(Transport::Udp, "0.0.0.0:3043")
        .unwrap();
    handler
        .network()
        .listen(Transport::Ws, "0.0.0.0:3044")
        .unwrap();

    // Read incoming network events.
    listener.for_each(move |event| match event {
        NodeEvent::Network(event) => {
            match event {
                NetEvent::Connected(_, _) => unreachable!(), // Used for explicit connections.
                NetEvent::Accepted(endpoint, _listener) => {
                    handle_network_connected(&mut handler, endpoint);
                }
                NetEvent::Message(endpoint, data) => {
                    let message: MessageFromWorker = bincode::deserialize(data).unwrap();
                    match message {
                        MessageFromWorker::Init => {
                            handler
                                .signals()
                                .send(NodeSignal::InitialiseWorker(endpoint));
                        }
                        MessageFromWorker::EpisodeCompleted(episode) => {
                            println!("Episode completed: {:?}", episode);
                        }
                    }
                }
                NetEvent::Disconnected(_endpoint) => println!("Client disconnected"), //Tcp or Ws
            }
        }
        NodeEvent::Signal(signal) => {
            match signal {
                NodeSignal::InitialiseWorker(endpoint) => {
                    println!("Initialising worker");
                    let reply = MessageFromLearner::InitialiseWorker {
                        seed: 1234,
                        block_size: PARAMETER_COUNT,
                        block_count: NOISE_BLOCKS,
                        parameter_count: PARAMETER_COUNT,
                    };
                    let response = bincode::serialize(&reply).unwrap();
                    handler.network().send(endpoint, response.as_slice());

                    if active_transfers.get_mut(&endpoint).is_none() {
                        active_transfers.insert(
                            endpoint,
                            ModelTransfer {
                                next_block_offset: 0,
                            },
                        );
                        handler
                            .signals()
                            .send(NodeSignal::NextTransferBlock(endpoint));
                    }
                }
                NodeSignal::NextTransferBlock(endpoint) => {
                    let transfer = active_transfers.get_mut(&endpoint).unwrap();
                    // All size units are relative to count of f32 floats.
                    let max_chunk_size = MAX_F32_CHUNK_SIZE;
                    let chunk_offset = transfer.next_block_offset;
                    let total_transfer_size = PARAMETER_COUNT;
                    let chunk_size = if chunk_offset + max_chunk_size > total_transfer_size {
                        total_transfer_size - chunk_offset
                    } else {
                        max_chunk_size
                    };
                    let chunk =
                        test_model.as_slice()[chunk_offset..chunk_offset + chunk_size].to_vec();
                    let chunk_hash = 0;
                    let reply = MessageFromLearner::ParameterChunk {
                        parameter_version: 0,
                        chunk,
                        chunk_offset,
                        chunk_hash,
                    };
                    let response = bincode::serialize(&reply).unwrap();
                    handler.network().send(endpoint, response.as_slice());
                    transfer.next_block_offset += chunk_size;
                    if transfer.next_block_offset < total_transfer_size {
                        handler
                            .signals()
                            .send(NodeSignal::NextTransferBlock(endpoint));
                    } else {
                        active_transfers.remove(&endpoint);
                    }
                }
            }
        }
    });
}


fn handle_network_connected(
    handler: &mut node::NodeHandler<NodeSignal>,
    endpoint: Endpoint,
) {
    println!("Endpoint connection: {:?}", endpoint);
    handler
        .signals()
        .send(NodeSignal::NewConnectedWorker(endpoint));

    handler
        .signals()
        .send_with_timer(NodeSignal::WorkerInitialisationTimeout(endpoint), Duration::from_millis(
            WORKER_INITIALISATION_TIMEOUT_MS,
        ));
}

fn handle_network_disconnected(
    handler: &mut node::NodeHandler<NodeSignal>,
    endpoint: Endpoint,
    active_transfers: &mut FnvHashMap<Endpoint, ParameterTransfer>,
) {
    println!("Endpoint disconnection: {:?}", endpoint);
    active_transfers.remove(&endpoint);
}