use std::collections::HashMap;
use std::mem;

extern "C" {
    fn grad_math_matmul_backward(a: usize, b: usize, d_output: usize, allocator: *mut std::ffi::c_void);
    fn grad_math_add_backward(a: usize, b: usize, d_output: usize);
    fn grad_math_sub_backward(a: usize, b: usize, d_output: usize);
    fn grad_math_mul_backward(a: usize, b: usize, d_output: usize);
    fn grad_math_div_backward(a: usize, b: usize, d_output: usize);
    fn grad_math_scale_backward(tensor: usize, scalar: f32, d_output: usize);
    fn grad_math_add_scalar_backward(tensor: usize, d_output: usize);
    fn grad_math_transpose_backward(input: usize, d_output: usize, allocator: *mut std::ffi::c_void);
    fn grad_math_reshape_backward(input: usize, d_output: usize, allocator: *mut std::ffi::c_void);
    fn grad_math_sum_backward(tensor: usize, d_output: usize);
    fn grad_math_mean_backward(tensor: usize, d_output: usize);
    fn grad_math_dot_backward(a: usize, b: usize, d_output: usize);
    fn grad_math_add_bias_backward(matrix: usize, bias: usize, d_output: usize);
    fn grad_math_matvec_backward(matrix: usize, vec: usize, d_output: usize, allocator: *mut std::ffi::c_void);
    fn grad_math_outer_backward(a: usize, b: usize, d_output: usize);
    fn grad_math_frobenius_norm_backward(tensor: usize, d_output: usize);
    fn grad_math_clamp_backward(tensor: usize, min_val: f32, max_val: f32, d_output: usize);
    fn grad_math_abs_backward(tensor: usize, d_output: usize);
    fn grad_math_relu_backward(input: usize, d_output: usize);
    fn grad_math_sigmoid_backward(input: usize, d_output: usize);
    fn grad_math_tanh_backward(input: usize, d_output: usize);
    fn grad_math_elu_backward(input: usize, alpha: f32, d_output: usize);
    fn grad_math_leaky_relu_backward(input: usize, negative_slope: f32, d_output: usize);
    fn grad_math_gelu_backward(input: usize, d_output: usize);
    fn grad_math_softmax_backward(input: usize, softmax_output: usize, d_output: usize);
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    None, // Linear activation
}

/// Context passed to layers during forward pass
/// Automatically manages tape recording
pub struct ForwardContext<'a> {
    pub tape: &'a mut Tape,
    pub next_tensor_id: &'a mut usize,
}

impl<'a> ForwardContext<'a> {
    /// Register new tensor and return its ID
    pub fn register_output(&mut self, shape: Vec<usize>, dtype: String) -> usize {
        let id = *self.next_tensor_id;
        *self.next_tensor_id += 1;
        self.tape.register_tensor_with_id(id, shape, dtype);
        id
    }

    /// Record an operation on the tape
    pub fn record_op(
        &mut self,
        op: Operation,
        inputs: Vec<usize>,
        output: usize,
        metadata: OperationMetadata,
    ) {
        self.tape.push_node(op, inputs, output, metadata);
    }
}

/// Represents the type of operation performed in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    MatMul,
    Add,
    Sub,
    Mul,
    Div,
    Scale,
    AddScalar,
    Transpose,
    Reshape,
    Sum,
    Mean,
    Dot,
    AddBias,
    MatVec,
    Outer,
    FrobeniusNorm,
    Clamp,
    Abs,
    ReLU,
    Sigmoid,
    Tanh,
    ELU,
    LeakyReLU,
    GELU,
    Softmax,
    MSELoss,
    CrossEntropyLoss,
}

/// Metadata associated with specific operations
#[derive(Debug, Clone)]
pub enum OperationMetadata {
    None,
    ScaleFactor(f32),
    AddScalarValue(f32),
    ClampBounds { min: f32, max: f32 },
    ELUAlpha(f32),
    LeakyReLUSlope(f32),
    SoftmaxOutputShape(Vec<usize>),
}

/// A node in the computation graph representing one operation
#[derive(Debug)]
pub struct GradNode {
    /// The operation type
    pub op: Operation,
    /// Pointers to input tensors (owned by Zig)
    pub inputs: Vec<usize>, // Use indices into a tensor registry instead of raw pointers
    /// Index of output tensor
    pub output: usize,
    /// Additional metadata for the operation
    pub metadata: OperationMetadata,
}

/// The computation tape that records all operations during forward pass
#[derive(Debug)]
pub struct Tape {
    /// Ordered list of operations performed
    pub nodes: Vec<GradNode>,
    /// Registry mapping tensor IDs to their data (managed by Zig)
    pub tensor_registry: HashMap<usize, TensorInfo>,
    /// Counter for generating unique tensor IDs
    next_tensor_id: usize,
}

/// Information about a tensor (shape, etc.)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: String, // "f32", etc.
}

impl Tape {
    /// Create a new empty tape
    pub fn new() -> Self {
        Tape {
            nodes: Vec::new(),
            tensor_registry: HashMap::new(),
            next_tensor_id: 0,
        }
    }

    /// Register a tensor and get its ID
    pub fn register_tensor(&mut self, shape: Vec<usize>, dtype: String) -> usize {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        self.tensor_registry.insert(id, TensorInfo { shape, dtype });
        id
    }

    /// Register a tensor with a specific ID
    pub fn register_tensor_with_id(&mut self, id: usize, shape: Vec<usize>, dtype: String) {
        self.tensor_registry.insert(id, TensorInfo { shape, dtype });
        if id >= self.next_tensor_id {
            self.next_tensor_id = id + 1;
        }
    }

    /// Get next available tensor ID without registering
    pub fn get_next_id(&self) -> usize {
        self.next_tensor_id
    }

    /// Add an operation node to the tape
    pub fn push_node(&mut self, op: Operation, inputs: Vec<usize>, output: usize, metadata: OperationMetadata) {
        self.nodes.push(GradNode {
            op,
            inputs,
            output,
            metadata,
        });
    }

    /// Clear the tape after backward pass
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.tensor_registry.clear();
        self.next_tensor_id = 0;
    }

    /// Get the number of operations recorded
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if tape is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    /// Perform backward pass through the computation graph
    /// This traverses nodes in reverse order and calls appropriate backward functions
    pub fn backward(&self) {
        // Iterate through nodes in reverse order
        for node in self.nodes.iter().rev() {
            self.backward_node(node);
        }
    }

    /// Execute backward pass for a single node
    fn backward_node(&self, node: &GradNode) {
        match node.op {
            Operation::MatMul => {
                // Call Zig matmul_backward
                self.call_matmul_backward(node);
            }
            Operation::Add => {
                self.call_add_backward(node);
            }
            Operation::Sub => {
                self.call_sub_backward(node);
            }
            Operation::Mul => {
                self.call_mul_backward(node);
            }
            Operation::Div => {
                self.call_div_backward(node);
            }
            Operation::Scale => {
                if let OperationMetadata::ScaleFactor(scalar) = &node.metadata {
                    self.call_scale_backward(node, *scalar);
                }
            }
            Operation::AddScalar => {
                self.call_add_scalar_backward(node);
            }
            Operation::Transpose => {
                self.call_transpose_backward(node);
            }
            Operation::Reshape => {
                self.call_reshape_backward(node);
            }
            Operation::Sum => {
                self.call_sum_backward(node);
            }
            Operation::Mean => {
                self.call_mean_backward(node);
            }
            Operation::Dot => {
                self.call_dot_backward(node);
            }
            Operation::AddBias => {
                self.call_add_bias_backward(node);
            }
            Operation::MatVec => {
                self.call_matvec_backward(node);
            }
            Operation::Outer => {
                self.call_outer_backward(node);
            }
            Operation::FrobeniusNorm => {
                self.call_frobenius_norm_backward(node);
            }
            Operation::Clamp => {
                if let OperationMetadata::ClampBounds { min, max } = &node.metadata {
                    self.call_clamp_backward(node, *min, *max);
                }
            }
            Operation::Abs => {
                self.call_abs_backward(node);
            }
            Operation::ReLU => {
                self.call_relu_backward(node);
            }
            Operation::Sigmoid => {
                self.call_sigmoid_backward(node);
            }
            Operation::Tanh => {
                self.call_tanh_backward(node);
            }
            Operation::ELU => {
                if let OperationMetadata::ELUAlpha(alpha) = &node.metadata {
                    self.call_elu_backward(node, *alpha);
                }
            }
            Operation::LeakyReLU => {
                if let OperationMetadata::LeakyReLUSlope(slope) = &node.metadata {
                    self.call_leaky_relu_backward(node, *slope);
                }
            }
            Operation::GELU => {
                self.call_gelu_backward(node);
            }
            Operation::Softmax => {
                self.call_softmax_backward(node);
            }
            Operation::MSELoss | Operation::CrossEntropyLoss => {
                // Loss functions don't need backward calls - they initialize gradients
            }
        }
    }

    // Placeholder FFI calls to Zig backward functions
    // In a real implementation, these would be extern "C" functions calling into Zig

    fn call_matmul_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_matmul_backward(_node.inputs[0], _node.inputs[1], _node.output, std::ptr::null_mut());
        }
    }

    fn call_add_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_add_backward(_node.inputs[0], _node.inputs[1], _node.output);
        }
    }

    fn call_sub_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_sub_backward(_node.inputs[0], _node.inputs[1], _node.output);
        }
    }

    fn call_mul_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_mul_backward(_node.inputs[0], _node.inputs[1], _node.output);
        }
    }

    fn call_div_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_div_backward(_node.inputs[0], _node.inputs[1], _node.output);
        }
    }

    fn call_scale_backward(&self, _node: &GradNode, _scalar: f32) {
        unsafe {
            grad_math_scale_backward(_node.inputs[0], _scalar, _node.output);
        }
    }

    fn call_add_scalar_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_add_scalar_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_transpose_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_transpose_backward(_node.inputs[0], _node.output, std::ptr::null_mut());
        }
    }

    fn call_reshape_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_reshape_backward(_node.inputs[0], _node.output, std::ptr::null_mut());
        }
    }

    fn call_sum_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_sum_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_mean_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_mean_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_dot_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_dot_backward(_node.inputs[0], _node.inputs[1], _node.output);
        }
    }

    fn call_add_bias_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_add_bias_backward(_node.inputs[0], _node.inputs[1], _node.output);
        }
    }

    fn call_matvec_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_matvec_backward(_node.inputs[0], _node.inputs[1], _node.output, std::ptr::null_mut());
        }
    }

    fn call_outer_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_outer_backward(_node.inputs[0], _node.inputs[1], _node.output);
        }
    }

    fn call_frobenius_norm_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_frobenius_norm_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_clamp_backward(&self, _node: &GradNode, _min: f32, _max: f32) {
        unsafe {
            grad_math_clamp_backward(_node.inputs[0], _min, _max, _node.output);
        }
    }

    fn call_abs_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_abs_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_relu_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_relu_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_sigmoid_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_sigmoid_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_tanh_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_tanh_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_elu_backward(&self, _node: &GradNode, _alpha: f32) {
        unsafe {
            grad_math_elu_backward(_node.inputs[0], _alpha, _node.output);
        }
    }

    fn call_leaky_relu_backward(&self, _node: &GradNode, _slope: f32) {
        unsafe {
            grad_math_leaky_relu_backward(_node.inputs[0], _slope, _node.output);
        }
    }

    fn call_gelu_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_gelu_backward(_node.inputs[0], _node.output);
        }
    }

    fn call_softmax_backward(&self, _node: &GradNode) {
        unsafe {
            grad_math_softmax_backward(_node.inputs[0], _node.output, _node.output);
        }
    }
}

/// Example usage of the Tape for automatic differentiation
pub fn example_usage() {
    let mut tape = Tape::new();

    // Register some tensors
    let input_id = tape.register_tensor(vec![784], "f32".to_string());
    let weight_id = tape.register_tensor(vec![784, 128], "f32".to_string());
    let bias_id = tape.register_tensor(vec![128], "f32".to_string());
    let hidden_id = tape.register_tensor(vec![128], "f32".to_string());
    let output_id = tape.register_tensor(vec![10], "f32".to_string());

    // Forward pass - simulate operations
    // hidden = matmul(input, weight) + bias
    tape.push_node(Operation::MatMul, vec![input_id, weight_id], hidden_id, OperationMetadata::None);
    tape.push_node(Operation::AddBias, vec![hidden_id, bias_id], hidden_id, OperationMetadata::None);

    // hidden = relu(hidden)
    tape.push_node(Operation::ReLU, vec![hidden_id], hidden_id, OperationMetadata::None);

    // output = matmul(hidden, output_weight) + output_bias
    let output_weight_id = tape.register_tensor(vec![128, 10], "f32".to_string());
    let output_bias_id = tape.register_tensor(vec![10], "f32".to_string());
    tape.push_node(Operation::MatMul, vec![hidden_id, output_weight_id], output_id, OperationMetadata::None);
    tape.push_node(Operation::AddBias, vec![output_id, output_bias_id], output_id, OperationMetadata::None);

    // output = softmax(output)
    tape.push_node(Operation::Softmax, vec![output_id], output_id, OperationMetadata::SoftmaxOutputShape(vec![10]));

    // Compute loss
    let target_id = tape.register_tensor(vec![10], "f32".to_string());
    let loss_id = tape.register_tensor(vec![1], "f32".to_string());
    tape.push_node(Operation::CrossEntropyLoss, vec![output_id, target_id], loss_id, OperationMetadata::None);

    println!("Forward pass recorded {} operations", tape.len());

    // Backward pass
    tape.backward();

    println!("Backward pass completed");
}

// =============================================================================
// LAYER ABSTRACTION SYSTEM
// =============================================================================

/// Trait for neural network layers
pub trait Layer {
    /// Forward pass - automatically records operations on tape
    fn forward(&mut self, input_id: usize, ctx: &mut ForwardContext) -> Result<usize, String>;

    /// Get mutable references to learnable parameters (weights, biases)
    fn parameters(&mut self) -> Vec<usize>;

    /// Get layer name for debugging
    fn name(&self) -> &str;
}

/// Dense/Linear layer: output = activation(input @ W + b)
pub struct Linear {
    pub input_size: usize,
    pub output_size: usize,
    pub weights_id: usize,
    pub bias_id: usize,
    pub activation: Activation,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        Linear {
            input_size,
            output_size,
            weights_id: 0, // Will be set by model
            bias_id: 0,    // Will be set by model
            activation,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input_id: usize, ctx: &mut ForwardContext) -> Result<usize, String> {
        // MatMul: input @ weights
        let matmul_output = ctx.register_output(vec![1, self.output_size], "f32".to_string());
        ctx.record_op(
            Operation::MatMul,
            vec![input_id, self.weights_id],
            matmul_output,
            OperationMetadata::None,
        );

        // AddBias: matmul_output + bias
        let bias_output = ctx.register_output(vec![1, self.output_size], "f32".to_string());
        ctx.record_op(
            Operation::AddBias,
            vec![matmul_output, self.bias_id],
            bias_output,
            OperationMetadata::None,
        );

        // Activation
        let activation = match self.activation {
            Activation::None => return Ok(bias_output),
            Activation::ReLU => Operation::ReLU,
            Activation::Sigmoid => Operation::Sigmoid,
            Activation::Tanh => Operation::Tanh,
        };

        let activation_output = ctx.register_output(vec![1, self.output_size], "f32".to_string());
        ctx.record_op(
            activation,
            vec![bias_output],
            activation_output,
            OperationMetadata::None,
        );

        Ok(activation_output)
    }

    fn parameters(&mut self) -> Vec<usize> {
        vec![self.weights_id, self.bias_id]
    }

    fn name(&self) -> &str {
        "Linear"
    }
}

/// LSTM layer with all four gates
pub struct LSTM {
    pub input_size: usize,
    pub hidden_size: usize,
    // Input gate parameters
    pub w_ii: usize,
    pub w_hi: usize,
    pub b_i: usize,
    // Forget gate parameters
    pub w_if: usize,
    pub w_hf: usize,
    pub b_f: usize,
    // Cell gate parameters
    pub w_ic: usize,
    pub w_hc: usize,
    pub b_c: usize,
    // Output gate parameters
    pub w_io: usize,
    pub w_ho: usize,
    pub b_o: usize,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        LSTM {
            input_size,
            hidden_size,
            w_ii: 0,
            w_hi: 0,
            b_i: 0,
            w_if: 0,
            w_hf: 0,
            b_f: 0,
            w_ic: 0,
            w_hc: 0,
            b_c: 0,
            w_io: 0,
            w_ho: 0,
            b_o: 0,
        }
    }
}

impl Layer for LSTM {
    fn forward(&mut self, input_id: usize, ctx: &mut ForwardContext) -> Result<usize, String> {
        // Simplified LSTM: single timestep forward pass
        // For full LSTM, would need to process sequence and maintain state
        
        // Input gate: i = sigmoid(W_ii @ x + W_hi @ h + b_i)
        let input_gate_result = ctx.register_output(vec![1, self.hidden_size], "f32".to_string());
        ctx.record_op(
            Operation::MatMul,
            vec![input_id, self.w_ii],
            input_gate_result,
            OperationMetadata::None,
        );

        // For full implementation would add h contribution and apply sigmoid
        // This is a simplified version
        let activation_output = ctx.register_output(vec![1, self.hidden_size], "f32".to_string());
        ctx.record_op(
            Operation::Sigmoid,
            vec![input_gate_result],
            activation_output,
            OperationMetadata::None,
        );

        Ok(activation_output)
    }

    fn parameters(&mut self) -> Vec<usize> {
        vec![
            self.w_ii, self.w_hi, self.b_i, self.w_if, self.w_hf, self.b_f, self.w_ic, self.w_hc,
            self.b_c, self.w_io, self.w_ho, self.b_o,
        ]
    }

    fn name(&self) -> &str {
        "LSTM"
    }
}

// =============================================================================
// MODEL CONTAINER - Orchestrates layers and automatic tape management
// =============================================================================

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
    pub tape: Tape,
    pub next_tensor_id: usize,
}

impl Model {
    /// Create new model
    pub fn new() -> Self {
        Model {
            layers: Vec::new(),
            tape: Tape::new(),
            next_tensor_id: 0,
        }
    }

    /// Add a layer to the model
    pub fn add_layer(&mut self, mut layer: Box<dyn Layer>) -> Result<(), String> {
        // Get parameters and register them
        let param_ids = layer.parameters();
        for param_id in param_ids {
            self.tape.register_tensor_with_id(param_id, vec![1, 1], "f32".to_string());
        }
        self.layers.push(layer);
        Ok(())
    }

    /// Forward pass through all layers - automatically records to tape
    pub fn forward(&mut self, input_id: usize) -> Result<usize, String> {
        let mut current_id = input_id;

        for layer in &mut self.layers {
            let mut ctx = ForwardContext {
                tape: &mut self.tape,
                next_tensor_id: &mut self.next_tensor_id,
            };
            current_id = layer.forward(current_id, &mut ctx)?;
        }

        Ok(current_id)
    }

    /// Backward pass through computation graph
    pub fn backward(&mut self) {
        self.tape.backward();
    }

    /// Clear tape after training step
    pub fn clear_tape(&mut self) {
        self.tape.clear();
    }

    /// Get all learnable parameters from all layers
    pub fn get_parameters(&mut self) -> Vec<usize> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    /// Get layer by index
    pub fn get_layer(&mut self, idx: usize) -> Option<&mut Box<dyn Layer>> {
        self.layers.get_mut(idx)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
