Embedding is only one major component. The real goal is to build a complete neural-network language model that uses several previous characters to predict the next character. This is the step from the simple bigram model toward the basic architecture behind modern language models.

### Big Picture

In [name-generator-work: github](https://github.com/ignius299792458/name-generator), the model was essentially:

```text
previous 1 character
        ↓
count/probability table
        ↓
next character
```

This changes this to:

```text
previous 3 characters
        ↓
embedding vectors
        ↓
MLP neural network
        ↓
next-character probabilities
```

So instead of remembering only one previous character, the model now has a context window.

### 1. Build the training dataset

For a name like:

```text
emma
```

with `block_size = 3`, training examples look conceptually like:

```text
... → e
..e → m
.em → m
emm → a
mma → .
```

Therefore:

```python
X.shape = [N, 3]
Y.shape = [N]
```

`X` contains the three previous character IDs, and `Y` contains the character the model should predict.

### 2. Learn embeddings

Instead of feeding integer IDs such as:

```text
[0, 5, 13]
```

directly into the neural network, we create:

```python
C = torch.randn((27, embedding_dim))
```

`C` is the embedding table.

For `embedding_dim = 2`:

```text
C.shape = [27, 2]
```

Then:

```python
emb = C[X]
```

turns:

```text
[0, 5, 13]
```

into something like:

```text
[
  C[0],
  C[5],
  C[13]
]
```

Each character is now represented by a learned vector rather than just an integer.

This is one of the most important ideas in the Word_Generator, but it is not the entire model.

### 3. Combine the context

If there are three characters and every character has a 2-dimensional embedding:

```text
3 characters × 2 dimensions = 6 numbers
```

So:

```python
emb.shape
# [batch, 3, 2]
```

gets flattened:

```python
emb.view(-1, 6)
```

giving:

```text
[batch, 6]
```

Now the neural network can process the three-character context together.

### 4. Hidden layer

Then Andrej constructs an MLP:

```python
h = torch.tanh(
    emb.view(-1, 6) @ W1 + b1
)
```

For example:

```text
6 input features
      ↓
100 hidden neurons
```

so:

```python
W1.shape = [6, 100]
b1.shape = [100]
```

This hidden layer learns patterns involving combinations of characters.

For example, it might learn that some contexts such as:

```text
"mar"
"ell"
"ann"
```

have different likely continuations.

This is already substantially more expressive than a bigram count table.

### 5. Predict logits for all characters

The hidden layer is then projected to 27 possible output characters:

```python
logits = h @ W2 + b2
```

where:

```text
W2.shape = [100, 27]
b2.shape = [27]
```

For each training example:

```text
27 logits
```

are produced—one raw score for every possible next character.

### 6. Cross-entropy loss

Instead of manually calculating softmax probabilities and negative log-likelihood, PyTorch can do:

```python
loss = F.cross_entropy(logits, Y)
```

Conceptually this performs:

```text
logits
 ↓
softmax probabilities
 ↓
probability assigned to correct character
 ↓
negative log likelihood
 ↓
average loss
```

Lower loss means the model is assigning greater probability to the correct next characters.

### 7. Backpropagation trains everything

The trainable parameters are:

```python
parameters = [
    C,
    W1, b1,
    W2, b2
]
```

Backpropagation flows:

```text
loss
 ↓
W2, b2
 ↓
hidden layer
 ↓
W1, b1
 ↓
embeddings
 ↓
C
```

This last part is extremely important.

`C` starts random:

```python
C = torch.randn(...)
```

but because gradients reach `C`, the embedding vectors gradually move into useful positions.

So you are not manually designing the embedding space. The model discovers it through next-character prediction.

### 8. Minibatch training

Because the dataset contains hundreds of thousands of context-target examples, computing the entire dataset every iteration is inefficient.

Instead:

```python
ix = torch.randint(0, X_train.shape[0], (32,))
```

selects a minibatch.

Then training becomes:

```text
sample 32 examples
      ↓
forward pass
      ↓
loss
      ↓
backward
      ↓
update parameters
      ↓
repeat
```

This introduces the minibatch training procedure used extensively in deep learning.

### 9. Learning rate

The Word_Generator also investigates how large parameter updates should be:

```python
p -= learning_rate * p.grad
```

Too small:

```text
training progresses very slowly
```

Too large:

```text
loss becomes unstable or fails to converge
```

This introduces hyperparameter tuning rather than treating training as simply “run gradient descent.”

### 10. Train / dev / test split

Another major concept is separating data into:

```text
Training set  → learn parameters
Dev set       → tune architecture/hyperparameters
Test set      → final unbiased evaluation
```

For example:

```text
80% train
10% dev
10% test
```

This is where its starts teaching proper machine-learning methodology, not merely neural-network mathematics.

### 11. Model capacity

You can change:

```text
embedding dimension
hidden-layer size
context length
```

to increase or decrease model capacity.

For example:

```text
27 × 2 embedding
3-token context
100 hidden neurons
```

is a relatively small model.

Increasing these values gives the network more ability to represent patterns, but can also increase computation and overfitting risk.

### 12. Visualizing learned embeddings

Because Andrej initially uses a two-dimensional embedding:

```python
C.shape = [27, 2]
```

you can directly plot:

```python
plt.scatter(C[:,0], C[:,1])
```

and observe where characters move during training.

This provides physical intuition for what “learning representations” means.

Characters are no longer simply:

```text
a = 1
b = 2
c = 3
```

They become learned points in a continuous vector space.

### 13. Sampling from the trained model

Finally, after training:

```text
...
 ↓
predict next character
 ↓
sample
 ↓
shift context
 ↓
predict again
 ↓
repeat until "."
```

Example:

```text
[., ., .]
    ↓
e

[., ., e]
    ↓
m

[., e, m]
    ↓
m

[e, m, m]
    ↓
a
```

This generates entirely new names character-by-character.

### What Word_Generator Actually Teaches

The complete model is:

```text
characters
    ↓
integer token IDs
    ↓
context window
    ↓
embedding lookup C[X]
    ↓
flatten context embeddings
    ↓
Linear layer W1 + b1
    ↓
tanh
    ↓
Linear layer W2 + b2
    ↓
logits
    ↓
cross entropy
    ↓
backpropagation
    ↓
gradient descent
    ↓
learn C, W1, b1, W2, b2
```

So I would summarize Word_Generator in one sentence as:

> Word_Generator builds a neural language model where several previous characters are converted into learned embeddings, processed by an MLP, and trained with cross-entropy and backpropagation to predict the next character.

And the progression toward LLMs is:

```text
Bigram
current character
→ next character

MLP / Word_Generator
several previous characters
→ embeddings
→ MLP
→ next character

GPT
many previous tokens
→ embeddings
→ Transformer
→ next token
```

So the embedding table is one foundational innovation in Word_Generator, but the larger achievement is understanding the entire neural language-model training pipeline.
