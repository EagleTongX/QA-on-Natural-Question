import tensorflow as tf
from tensorflow.contrib import rnn
from attention import attention

class BiLSTM:
	def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, seq_length, lengths, is_training):
		"""
        :param embedded_chars: embedding input
        :param hidden_unit: how many hidden units in LSTM
        :param cell_type: only lstm/gru (RNN) at the moment
        :param num_layers: #RNN layers
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: how many different classes for the output 
        :param seq_length: max seq length
        :param lengths: actual length of the seq
        :param is_training: is training or testing
    	"""
		self.hidden_unit = hidden_unit
		self.dropout_rate = dropout_rate
		self.cell_type = cell_type
		self.num_layers = num_layers
		self.embedded_chars = embedded_chars
		self.initializers = initializers
		self.seq_length = seq_length
		self.num_labels = num_labels
		self.lengths = lengths
		self.embedding_dims = embedded_chars.shape[-1].value
		self.is_training = is_training

	def add_bilstm_layer(self):
		if self.is_training:
			self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

		with tf.variable_scope('rnn_layer'):
			cell_fw, cell_bw = self._bi_dir_rnn()
			if self.num_layers > 1:
				cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
				cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embedded_chars,
		                                                 dtype=tf.float32)
			outputs = tf.concat(outputs, axis=2)

		
		attn, alphas = attention(outputs)
		logits = self.project_to_desired_output_logits(outputs, name='lstm_logits')
			
		return logits, attn

	def _witch_cell(self):
		cell_tmp = None
		if self.cell_type == 'lstm':
			cell_tmp = rnn.LSTMCell(self.hidden_unit)
		elif self.cell_type == 'gru':
			cell_tmp = rnn.GRUCell(self.hidden_unit)
		return cell_tmp

	def _bi_dir_rnn(self):
		cell_fw = self._witch_cell()
		cell_bw = self._witch_cell()
		if self.dropout_rate is not None:
			cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
			cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
		return cell_fw, cell_bw


	def project_to_desired_output_logits(self, lstm_outputs, name=None):
		"""
		hidden layer between lstm layer and logits
		:param lstm_outputs: [batch_size, num_steps/sep_len, emb_size]
		:return: [batch_size, num_steps/sep_len, num_classes]
		"""
		with tf.variable_scope("project" if not name else name):
			with tf.variable_scope("hidden"):
				W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
		                            dtype=tf.float32, initializer=self.initializers.xavier_initializer())
				b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
		                            initializer=tf.zeros_initializer())
				output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
				hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

			with tf.variable_scope("logits"):
				W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
		                            dtype=tf.float32, initializer=self.initializers.xavier_initializer())
				b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
		                            initializer=tf.zeros_initializer())

				pred = tf.nn.xw_plus_b(hidden, W, b)
			return tf.reshape(pred, [-1, self.seq_length, self.num_labels])