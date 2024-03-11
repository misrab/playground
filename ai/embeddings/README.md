## summary

Embeddings represent members of a finite set (e.g. a vocabulary of words) as vectors. In essence, we want to take a discrete set of items and throw them into a geometric space. Presumably, imposing geometry means items can cluster together in groups based on some sort of similarity, such as semantic similarity with words. This helps us reduce the dimensionality of the problem: instead of one dimension for each item in the set (i.e. a million English words[^1]), we have a lower number for the embeddings (i.e. thousands for ChatGPT[^2]). In essence, we're saying that each item is made up of a smaller number of ingredients.

Why is a lower dimension useful? Each extra dimension loosely speaking adds an exponential amount of potential combinations of the data, also known as the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). This means that if we're trying to estimate some lower dimensional manifold in the space where the data sits (so that we can do things like prediction), we need exponentially more data to get a good statistical estimate of the manifold in any given region.

![manifold](/images/manifold.png)

## dev env

`pyenv versions`
`pyenv virtualenvs`
`pyenv activate [env]` // using 3.11.3 as a base
`pip freeze -r requirements.txt`

## Footnotes

[^1]: https://www.merriam-webster.com/help/faq-how-many-english-words
[^2]: https://openai.com/blog/new-and-improved-embedding-model
