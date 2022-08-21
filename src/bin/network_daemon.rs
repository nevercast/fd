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
use fnv::FnvHashMap;
use message_io::network::{Endpoint, NetEvent, Transport};
use message_io::node::{self, NodeEvent};

type Handler = node::NodeHandler<NodeSignal>;

const WORKER_INITIALISATION_TIMEOUT_MS: u64 = 3000;
const PARAMETER_COUNT: usize = 1_000_000;

// MAXIMUM MODEL AGE also applies to dropping received returns from worker episodes.
#[allow(dead_code)]
const MAXIMUM_MODEL_AGE: u32 = 10;

struct ConnectedWorker {
    has_initialised: bool,
}

enum NodeSignal {
    NewConnectedWorker(Endpoint),
    WorkerCheckTimeout(Endpoint),
    WorkerHasTimedOut(Endpoint),
    NextTransferBlock(Endpoint),
    InitialiseWorker(Endpoint),
    SendModelToWorker(Endpoint, ModelVersion),
    CleanupWorker(Endpoint),
}

fn main() {
    let latest_model_version: ModelVersion = 0;
    let _models = FnvHashMap::<ModelVersion, Arc<Vec<f32>>>::default();
    let mut active_transfers = FnvHashMap::<Endpoint, ModelTransfer>::default();
    let mut connected_workers = FnvHashMap::<Endpoint, ConnectedWorker>::default();

    let mut model = Vec::<f32>::with_capacity(PARAMETER_COUNT);
    for i in 0..PARAMETER_COUNT {
        model.push(i as f32);
    }

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
                    handle_network_connected(&handler, endpoint, &mut connected_workers);
                }
                NetEvent::Message(endpoint, data) => {
                    let message = deserialise_worker_message(data);
                    handle_worker_message(&handler, endpoint, message);
                }
                NetEvent::Disconnected(endpoint) => {
                    handle_network_disconnected(&handler, endpoint);
                }
            }
        }
        NodeEvent::Signal(signal) => match signal {
            NodeSignal::NewConnectedWorker(endpoint) => {
                handle_new_connected_worker(&handler, endpoint, &mut connected_workers);
            }
            NodeSignal::InitialiseWorker(endpoint) => {
                handle_worker_initialisation(
                    &handler,
                    endpoint,
                    latest_model_version,
                    &mut active_transfers,
                );
            }
            NodeSignal::SendModelToWorker(endpoint, latest_version) => {
                begin_model_transfer_if_required(
                    &handler,
                    endpoint,
                    latest_version,
                    &mut active_transfers,
                );
            }
            NodeSignal::WorkerCheckTimeout(endpoint) => {
                handle_worker_timeout_check(&handler, endpoint, &connected_workers);
            }
            NodeSignal::WorkerHasTimedOut(endpoint) => {
                handle_worker_timed_out(&handler, endpoint, &connected_workers);
            }
            NodeSignal::CleanupWorker(endpoint) => {
                handle_worker_cleanup(
                    &handler,
                    endpoint,
                    &mut connected_workers,
                    &mut active_transfers,
                );
            }
            NodeSignal::NextTransferBlock(endpoint) => {
                handle_next_transfer_block(&handler, endpoint, &mut active_transfers);
            }
        },
    });
}

fn handle_new_connected_worker(
    _handler: &node::NodeHandler<NodeSignal>,
    _endpoint: Endpoint,
    _connected_workers: &mut std::collections::HashMap<
        Endpoint,
        ConnectedWorker,
        std::hash::BuildHasherDefault<fnv::FnvHasher>,
    >,
) {
    todo!()
}

fn handle_next_transfer_block(
    _handler: &node::NodeHandler<NodeSignal>,
    _endpoint: Endpoint,
    _active_transfers: &mut std::collections::HashMap<
        Endpoint,
        ModelTransfer,
        std::hash::BuildHasherDefault<fnv::FnvHasher>,
    >,
) {
    todo!()
}

fn handle_worker_cleanup(
    _handler: &Handler,
    endpoint: Endpoint,
    connected_workers: &mut FnvHashMap<Endpoint, ConnectedWorker>,
    active_transfers: &mut FnvHashMap<Endpoint, ModelTransfer>,
) {
    println!("Worker {} is being cleaned up.", endpoint);
    connected_workers.remove(&endpoint);
    active_transfers.remove(&endpoint);
    println!("Worker {} cleaned up.", endpoint);
}

fn handle_worker_timed_out(
    handler: &node::NodeHandler<NodeSignal>,
    endpoint: Endpoint,
    _connected_workers: &FnvHashMap<Endpoint, ConnectedWorker>,
) {
    println!("Worker {} has timed out.", endpoint);
    handler
        .signals()
        .send_with_priority(NodeSignal::CleanupWorker(endpoint));
    // TODO test that this does as expected
    // It could do weird things for UDP, if we support it then this timeout might need to change
    handler.network().remove(endpoint.resource_id());
}

fn handle_worker_timeout_check(
    handler: &Handler,
    endpoint: Endpoint,
    connected_workers: &FnvHashMap<Endpoint, ConnectedWorker>,
) {
    if let Some(worker) = connected_workers.get(&endpoint) {
        if worker.has_initialised {
            return;
        }
        handler
            .signals()
            .send(NodeSignal::WorkerHasTimedOut(endpoint));
    }
}

fn begin_model_transfer_if_required(
    handler: &Handler,
    endpoint: Endpoint,
    latest_version: ModelVersion,
    active_transfers: &mut FnvHashMap<Endpoint, ModelTransfer>,
) {
    if active_transfers.get_mut(&endpoint).is_none() {
        active_transfers.insert(
            endpoint,
            ModelTransfer {
                model_version: latest_version,
                transfer_offset: 0,
                buffer: vec![0.0; 0],
            },
        );
        handler
            .signals()
            .send(NodeSignal::NextTransferBlock(endpoint));
    }
}

fn send_initialise_worker_message(handler: &Handler, endpoint: Endpoint) {
    let message = MessageFromLearner::InitialiseWorker {
        parameter_count: PARAMETER_COUNT,
    };
    let data = serialize_worker_response(message);
    handler.network().send(endpoint, data.as_slice());
}

fn handle_worker_initialisation(
    handler: &Handler,
    endpoint: Endpoint,
    latest_model_version: ModelVersion,
    _active_transfers: &mut FnvHashMap<Endpoint, ModelTransfer>,
) {
    println!("Initialising worker");
    send_initialise_worker_message(handler, endpoint);
    handler.signals().send(NodeSignal::SendModelToWorker(
        endpoint,
        latest_model_version,
    ));
}

fn deserialise_worker_message(data: &[u8]) -> MessageFromWorker {
    bincode::deserialize(data).unwrap()
}

fn serialize_worker_response(response: MessageFromLearner) -> Vec<u8> {
    bincode::serialize(&response).unwrap()
}

fn handle_worker_message(handler: &Handler, endpoint: Endpoint, message: MessageFromWorker) {
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

fn handle_network_connected(
    handler: &Handler,
    endpoint: Endpoint,
    connected_workers: &mut FnvHashMap<Endpoint, ConnectedWorker>,
) {
    println!("Endpoint connection: {:?}", endpoint);
    connected_workers.insert(
        endpoint,
        ConnectedWorker {
            has_initialised: false,
        },
    );

    handler
        .signals()
        .send(NodeSignal::NewConnectedWorker(endpoint));

    handler.signals().send_with_timer(
        NodeSignal::WorkerCheckTimeout(endpoint),
        Duration::from_millis(WORKER_INITIALISATION_TIMEOUT_MS),
    );
}

fn handle_network_disconnected(handler: &node::NodeHandler<NodeSignal>, endpoint: Endpoint) {
    println!("Endpoint disconnection: {:?}", endpoint);
    handler
        .signals()
        .send_with_priority(NodeSignal::CleanupWorker(endpoint));
}
