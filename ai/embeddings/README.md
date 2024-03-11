## Summary

### What and why

Embeddings represent members of a finite set (e.g. a vocabulary of words) as vectors. In essence, we want to take a discrete set of items and throw them into a geometric space. Presumably, imposing geometry means items can cluster together in groups based on some sort of similarity, such as semantic similarity with words. This helps us reduce the dimensionality of the problem: instead of one dimension for each item in the set (i.e. a million English words[^1]), we have a lower number for the embeddings (i.e. thousands for ChatGPT[^2]). In essence, we're saying that each item is made up of a smaller number of ingredients.

Why is a lower dimension useful? Each extra dimension loosely speaking adds an exponential amount of potential combinations of the data, also known as the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). This means that if we're trying to estimate some lower dimensional manifold in the space where the data sits (so that we can do things like prediction), we need exponentially more data to get a good statistical estimate of the manifold in any given region.

![manifold](./images/manifold.png)
_A manifold of data in a higher dimensional space_

So I'd say dimensionality reduction is biggest reason behind embeddings. Models should simply perform better with them. But there are other reasons they are useful. They may be geometrically interpretable[^3]. They can also take different types of input (e.g. images, audio, text) and project them into a common space.

### How

Embeddings are found by simply one-hot encoding a discrete set of items, and using a linear first layer in the neural net. If a one-hot encoding in position $k$ is denoted as the basis vector $e_k$, and the first linear layer of the neural net has weight matrix $W$ of size (embedding dimensions, input dimensions), we see that

$$
W \cdot e_k
$$

simply pulls out column $k$ of the weight matrix. Thus in this form, column $k$ of $W$ is simply the embedding for that item. I have seen places where this is represented as

$$
e_k^T \cdot \tilde{W}
$$

In this case, the weight matrix $\tilde{W}$ would have dimensions (input dimensions, embedding dimensions), and in fact $e_k^T$ would be pulling out a _row_.

Training the "embeddings" then simply means training the whole neural network as per usual, and simply interpretating the first linear layer's weights as the embeddings[^4].

### Musings

## Footnotes

[^1]: https://www.merriam-webster.com/help/faq-how-many-english-words
[^2]: https://openai.com/blog/new-and-improved-embedding-model
[^3]: https://www.sci.utah.edu/~beiwang/publications/Word_Embeddings_BeiWang_2017.pdf
[^4]: https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture
