---
author: Luca Lombardo
pubDatetime: 2026-08-14T00:00:00Z
title: Compression
slug: compression
featured: true
draft: false
description: compression
---

Over the past few weeks, I have repeatedly encountered the same claim on Hacker News: **compression is prediction**. An [ngrok article](https://ngrok.com/blog/compression-is-prediction) develops the idea through entropy coding and language models, while [Salvatore Sanfilippo](https://www.youtube.com/watch?v=UgRiVUce9sY) recently discussed the similarities and differences between predicting and compressing. Both start from a real mathematical connection. A model assigns probabilities to possible continuations, and an entropy coder turns those probabilities into shorter or longer representations.

I have spent the last few years working on compression, information theory, and compressed representations. I agree with the mathematics behind the claim, but as a general account of compression it seems to begin halfway through the story. Before assigning probabilities or predicting the next symbol, we must decide what the object is, which alternatives the decoder already knows to be possible, and what information the representation must preserve.

Compression can be defined before introducing a probability distribution. If an object belongs to a finite family, then representing it means distinguishing it from every other member of that family. The cardinality of the family already determines a lower bound on the number of bits required. If we consider an individual object instead, its shortest effective description leads to Kolmogorov complexity. Neither viewpoint requires a notion of the next symbol.

Probability becomes relevant when the possible objects should not all receive descriptions of the same length. A distribution tells us which objects may be assigned shorter descriptions and which ones must receive longer descriptions in exchange. When an object is generated sequentially, the probability of the whole object can then be factored into conditional probabilities for its successive symbols. This is the point at which lossless compression and prediction meet.

The interesting question is therefore not whether compression *is* prediction in every useful sense. It is what must already have been fixed before that equivalence becomes true, what notion of prediction it requires, and which parts of the compression problem remain outside it.


## Table of contents

## Compression Before Probability

Compression begins with descriptions. Given an object $x$, we seek a binary string from which a decoder can recover $x$ without ambiguity. A representation is shorter when it identifies the same object using fewer bits, but its length is meaningful only relative to what the encoder and decoder have already agreed upon. The description language, the class of admissible objects, and any information available to both sides determine what still needs to be encoded.

Kolmogorov complexity captures the most general version of this idea. After fixing a universal machine $U$, a program for $U$ acts as a description, and the Kolmogorov complexity of a binary string $x$ is

$$
K_U(x) = \min \{ |p| : U(p) = x \}
$$

Thus, $K_U(x)$ is the length of the shortest program that produces $x$. Any regularity that can be expressed algorithmically may contribute to a shorter description, independently of whether that regularity was expressed as a probability distribution. The definition concerns the description of a complete individual object, not the prediction of one of its symbols from the preceding ones.

The choice of $U$ affects the value of $K_U(x)$, but only by an additive constant independent of $x$. This makes Kolmogorov complexity a mathematically stable notion of ultimate description length. It does not, however, provide a compression algorithm. Kolmogorov complexity is not computable, so no procedure can determine the length of the shortest program for every string, much less construct that program.

Practical compression therefore replaces the search over all possible programs with restrictions that can be described and exploited effectively. One such restriction is the knowledge that the object belongs to a finite family $\mathcal{F}$. Once $\mathcal{F}$ is fixed, a representation must distinguish each member of $\mathcal{F}$ from every other member.

Consider a fixed-length encoding

$$
C : \mathcal{F} \longrightarrow \{0,1\}^{\ell}
$$

Lossless decoding requires $C$ to be injective. Since there are only $2^\ell$ binary strings of length $\ell$, injectivity implies

$$
2^\ell \geq |\mathcal{F}|
$$

Consequently,

$$
\ell \geq \left\lceil \log_2 |\mathcal{F}| \right\rceil
$$

Conversely, the members of $\mathcal{F}$ can be indexed and their indices represented in binary, so this bound can be attained. The quantity

$$
H_{\mathrm{wc}}(\mathcal{F}) = \log_2 |\mathcal{F}|
$$

is the *worst-case entropy* of the family. It measures the information required to identify an arbitrary member of $\mathcal{F}$ when no member is allowed to remain indistinguishable from another.

No probability distribution appears in this argument. The bound follows from the number of admissible objects and the number of distinct binary representations. Although $\log_2 |\mathcal{F}|$ is also the Shannon entropy of a uniform random variable over $\mathcal{F}$, the counting argument does not assume that objects are sampled uniformly. In fact, it does not assume that they are sampled at all.

The family $\mathcal{F}$ is part of the problem specification. If the decoder knows only that the object belongs to a larger family $\mathcal{G}$, then the representation must distinguish among the members of $\mathcal{G}$, and the corresponding bound becomes $\log_2 |\mathcal{G}|$. A smaller description is possible only when the restriction from $\mathcal{G}$ to $\mathcal{F}$ is already known or is itself communicated to the decoder. What counts as redundancy therefore depends on which alternatives the representation is required to distinguish.

Worst-case entropy treats all members of $\mathcal{F}$ symmetrically. It determines the length required when every object must fit within the same bound, but it cannot express that some objects occur more frequently than others. To exploit such an imbalance, a code must assign different lengths to different objects. Determining which descriptions should be shorter requires a probability distribution, and this changes the objective from worst-case length to expected length.

## When Possibilities Have Different Probabilities

Worst-case entropy assigns the same length to every member of $\mathcal{F}$. This guarantees that every possible object can be represented within $\lceil\log_2|\mathcal{F}|\rceil$ bits, but it cannot exploit differences in how often those objects occur. If some members appear more frequently than others, their shorter descriptions may compensate for longer descriptions assigned to rarer members.

To express these frequencies, we replace the family $\mathcal{F}$ with a source whose possible outputs form a finite set $\mathcal{X}$. For each $x\in\mathcal{X}$, the source specifies a probability

$$
P(x)=\Pr(X=x)
$$

where $P(x)>0$ and

$$
\sum_{x\in\mathcal{X}}P(x)=1
$$

A probability must now be translated into a quantity measured in bits. Since the probability of two independent outcomes is the product of their individual probabilities, the corresponding bit costs should add. The logarithm performs exactly this conversion. The [information content](https://en.wikipedia.org/wiki/Information_content) of an outcome $x$ is

$$
I_P(x)=-\log_2P(x)
$$

More probable outcomes have smaller information content. If an event has probability $2^{-b}$, observing it provides $b$ bits of information according to this definition.

Before the source produces an outcome, its information content is not known. Its average value is

$$
H(P) = \sum_{x\in\mathcal{X}}P(x)I_P(x) = -\sum_{x\in\mathcal{X}}P(x)\log_2P(x)
$$

This quantity is the [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_%28information_theory%29) of the source. It depends on the distribution $P$, rather than on the particular names assigned to the objects in $\mathcal{X}$. When $P$ is uniform, every object has information content $\log_2|\mathcal{X}|$, and Shannon entropy reduces to the worst-case entropy introduced in the previous section.

The values $I_P(x)$ describe ideal bit costs derived from the source probabilities. To obtain a representation, we need a code

$$
C:\mathcal{X}\longrightarrow\{0,1\}^*
$$

where $\{0,1\}^*$ is the set of all finite binary strings. The code assigns a codeword $C(x)$ to every outcome $x$, with length

$$
\ell_C(x)=|C(x)|
$$

Since the source produces $x$ with probability $P(x)$, the average number of bits used by the code is

$$
L_P(C) = \sum_{x\in\mathcal{X}}P(x)\ell_C(x)
$$

The entropy $H(P)$ averages the real-valued quantities $-\log_2P(x)$ determined by the source. The expected code length $L_P(C)$ averages the integer lengths of the binary strings chosen by the encoder. Whether these two averages can coincide depends on which collections of integer lengths can belong to a decodable code.

Assigning different strings to different objects is sufficient when each codeword is presented in isolation. It is not sufficient when codewords are concatenated. The same binary string could then admit two decompositions into codewords and hence two possible source sequences. A code is [uniquely decodable](https://en.wikipedia.org/wiki/Variable-length_code) when every concatenation of codewords has only one decomposition.

A [prefix code](https://en.wikipedia.org/wiki/Prefix_code) prevents this ambiguity by requiring that no codeword be a prefix of another. Its codewords can be placed at leaves of a binary tree, where each edge contributes one bit and the depth of a leaf equals the length of its codeword.

Let $m$ be at least as large as the longest codeword. A codeword of length $\ell_C(x)$ has $2^{m-\ell_C(x)}$ descendants at depth $m$. Because no codeword is a prefix of another, the sets of descendants belonging to different codewords are disjoint. The tree contains only $2^m$ nodes at that depth, so

$$
\sum_{x\in\mathcal{X}}2^{m-\ell_C(x)} \leq 2^m
$$

Dividing by $2^m$ gives

$$
\sum_{x\in\mathcal{X}}2^{-\ell_C(x)} \leq 1
$$

This is the [Kraft–McMillan inequality](https://en.wikipedia.org/wiki/Kraft%E2%80%93McMillan_inequality). The tree argument proves the inequality for prefix codes. The same inequality is necessary for every uniquely decodable code, even when its codewords do not form leaves of a prefix tree. Conversely, any collection of non-negative integer lengths satisfying the inequality can be realized by a prefix code.

The source probabilities obey a closely related identity. From the definition of information content,

$$
2^{-I_P(x)}=P(x)
$$

and therefore

$$
\sum_{x\in\mathcal{X}}2^{-I_P(x)} = \sum_{x\in\mathcal{X}}P(x) = 1
$$

The ideal lengths $I_P(x)$ satisfy the same constraint as the lengths of a prefix code, with equality. They may nevertheless fail to define a binary code because they need not be integers. We still need to determine how much this integrality constraint separates the entropy of the source from the expected length of an actual code.

First consider an arbitrary uniquely decodable code $C$. Define its Kraft sum by

$$
S_C = \sum_{x\in\mathcal{X}}2^{-\ell_C(x)}
$$

The Kraft–McMillan inequality gives $S_C\leq1$. The quantities $2^{-\ell_C(x)}$ may therefore sum to less than one, but normalization turns them into a probability distribution:

$$
Q_C(x) = \frac{2^{-\ell_C(x)}}{S_C}
$$

Solving this equation for the codeword length gives

$$
\ell_C(x) = -\log_2Q_C(x)-\log_2S_C
$$

Averaging over the source distribution produces

$$
L_P(C) = -\sum_{x\in\mathcal{X}}P(x)\log_2Q_C(x) -\log_2S_C
$$

Subtracting the entropy gives

$$
L_P(C)-H(P) = \sum_{x\in\mathcal{X}} P(x)\log_2\frac{P(x)}{Q_C(x)} -\log_2S_C
$$

The second term is non-negative because $S_C\leq1$. The first is also non-negative. Indeed, the concavity of the logarithm gives

$$
\begin{aligned}
\sum_{x\in\mathcal{X}} P(x)\log_2\frac{Q_C(x)}{P(x)} &\leq \log_2 \left( \sum_{x\in\mathcal{X}} P(x)\frac{Q_C(x)}{P(x)} \right) \\
&= \log_2 \left( \sum_{x\in\mathcal{X}} Q_C(x) \right) \\
&= 0
\end{aligned}
$$

Changing the sign yields

$$
\sum_{x\in\mathcal{X}} P(x)\log_2\frac{P(x)}{Q_C(x)} \geq 0
$$

It follows that every uniquely decodable code satisfies

$$
L_P(C)\geq H(P)
$$

Entropy is therefore a lower bound on the expected length of an actual code, rather than merely the average of a convenient assignment of real-valued lengths.

It remains to determine whether the lower bound can be approached. Rounding each ideal length upward gives the integers

$$
\ell(x) = \left\lceil-\log_2P(x)\right\rceil
$$

Since $\ell(x)\geq-\log_2P(x)$,

$$
2^{-\ell(x)} \leq P(x)
$$

Summing over $\mathcal{X}$ gives

$$
\sum_{x\in\mathcal{X}}2^{-\ell(x)} \leq \sum_{x\in\mathcal{X}}P(x) = 1
$$

The rounded lengths satisfy the Kraft–McMillan inequality, so a prefix code with those lengths exists. They also satisfy

$$
-\log_2P(x) \leq \ell(x) < -\log_2P(x)+1
$$

Averaging over the source gives

$$
H(P) \leq L_P(C) < H(P)+1
$$

The lower bound and the construction together form [Shannon’s Source Coding Theorem](https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem). For a known source, no uniquely decodable binary code has expected length below its entropy, while a prefix code can always remain within one bit of it. The gap comes from rounding real-valued information contents to integer codeword lengths.

Encoding several outcomes together can distribute this rounding cost over the whole group, making the additional cost per outcome arbitrarily small. This gives entropy its operational interpretation as the limiting average number of bits required by a known source. The argument still treats each encoded object, including any group of outcomes, as one value $x$ with one probability $P(x)$. Prediction appears only after we expose the internal order of a sequence and express its probability through a succession of conditional probabilities.


## When Compression Becomes Prediction

The previous section treated an object $x$ as a single outcome with probability $P(x)$. A sequence $x_{1:n}=x_1,\ldots,x_n$ can be treated in the same way, but its order permits a second description. Instead of assigning one probability directly to the complete sequence, we can assign a probability to each symbol after observing the prefix that precedes it.

Write

$$
x_{<i}=x_1,\ldots,x_{i-1}
$$

for the prefix before position $i$. For every prefix with positive probability, the conditional probability of the next symbol is

$$
P(x_i\mid x_{<i}) = \frac{P(x_{1:i})}{P(x_{<i})}
$$

For $i=1$, the prefix is empty and $P(x_1\mid x_{<1})$ is simply $P(x_1)$. Multiplying the conditional probabilities makes the intermediate prefix probabilities cancel:

$$
\begin{aligned}
\prod_{i=1}^n P(x_i\mid x_{<i}) &= P(x_1) \frac{P(x_{1:2})}{P(x_1)} \frac{P(x_{1:3})}{P(x_{1:2})} \cdots \frac{P(x_{1:n})}{P(x_{<n})} \\
&= P(x_{1:n})
\end{aligned}
$$

This identity is the [chain rule for probability](https://en.wikipedia.org/wiki/Chain_rule_%28probability%29):

$$
P(x_{1:n}) = \prod_{i=1}^n P(x_i\mid x_{<i})
$$

No independence assumption is involved. Each conditional distribution may depend on the entire preceding sequence. The factorization only uses the fact that the object has been presented as an ordered sequence.

Applying $-\log_2$ turns the product into a sum:

$$
-\log_2P(x_{1:n}) = \sum_{i=1}^n -\log_2P(x_i\mid x_{<i})
$$

The information content of the complete sequence is therefore the sum of the information contents of its symbols under their respective conditional distributions. A symbol may be expensive at one position and cheap at another because its probability changes with the prefix.

Averaging the same identity over all possible sequences gives

$$
H(X_{1:n}) = \sum_{i=1}^n H(X_i\mid X_{<i})
$$

where the [conditional entropy](https://en.wikipedia.org/wiki/Conditional_entropy)

$$
H(X_i\mid X_{<i}) = \mathbb{E}\left[-\log_2P(X_i\mid X_{<i})\right]
$$

measures the average information in the next symbol once the preceding symbols are known. The entropy of the whole sequence is not generally the sum of the separate entropies $H(X_i)$. It is the sum of the information that remains at each position after the available context has been taken into account.

A compressor rarely has direct access to the true conditional probabilities. It instead uses a model $Q$ that, for every prefix $x_{<i}$, produces a distribution

$$
Q(\cdot\mid x_{<i})
$$

over the next symbol. The output must be a complete distribution, with

$$
\sum_{a\in\mathcal{X}}Q(a\mid x_{<i})=1
$$

Returning only the most likely symbol would not provide enough information for compression. The encoder may encounter any symbol in $\mathcal{X}$ and needs a bit cost for whichever symbol actually occurs. Two models may select the same most likely symbol while assigning very different probabilities to the observed one, and those probabilities produce different encoded lengths.

The conditional distributions supplied by $Q$ define a probability for the complete sequence:

$$
Q(x_{1:n}) = \prod_{i=1}^n Q(x_i\mid x_{<i})
$$

The cost assigned by the model to the observed sequence is

$$
\begin{aligned}
\mathcal{L}_Q(x_{1:n}) &= \sum_{i=1}^n -\log_2Q(x_i\mid x_{<i}) \\
&= -\log_2Q(x_{1:n})
\end{aligned}
$$

This is the model’s [logarithmic loss](https://en.wikipedia.org/wiki/Scoring_rule), measured in bits. A confident probability assigned to the observed symbol produces a small loss. A small probability produces a large loss. Assigning probability zero produces infinite loss, so a model used for lossless compression must assign positive probability to every symbol that may occur.

During generation, a model may choose or sample a symbol from its next-symbol distribution. During compression, the encoder already knows the actual next symbol. It asks the model for the distribution, observes the probability assigned to the actual symbol, and uses that probability to encode it. Prediction here refers to constructing the distribution, not to replacing the next symbol with a guess.

The model still does not produce a bitstream. It produces the probabilities from which ideal bit lengths can be computed. A separate coding procedure must turn those probabilities into a uniquely decodable representation. This separates statistical coding into two operations. The model determines how much probability each possible continuation receives, while the entropy coder converts those assignments into bits.

[Arithmetic coding](https://en.wikipedia.org/wiki/Arithmetic_coding) makes this conversion directly. It begins with the interval $[0,1)$. Before encoding position $i$, the model supplies the distribution $Q(\cdot\mid x_{<i})$. The current interval is divided into adjacent subintervals whose widths are proportional to these probabilities, and the encoder retains the subinterval assigned to the actual symbol $x_i$.

If the current interval has width $w_{i-1}$, the selected subinterval has width

$$
w_i = w_{i-1}Q(x_i\mid x_{<i})
$$

Starting from $w_0=1$, repeated subdivision gives

$$
\begin{aligned}
w_n &= \prod_{i=1}^n Q(x_i\mid x_{<i}) \\
&= Q(x_{1:n})
\end{aligned}
$$

The final interval identifies the complete sequence. A sufficiently precise binary fraction inside that interval identifies it to the decoder, and an interval of width $w_n$ requires approximately

$$
-\log_2w_n = -\log_2Q(x_{1:n})
$$

bits to specify. Arithmetic coding reaches this length within a constant number of bits, without requiring an integer codeword length for each individual symbol. This is why its total length follows the sum of the conditional information contents rather than the sum of separately rounded symbol lengths.

Decoding repeats the interval subdivisions. After recovering the prefix $x_{<i}$, the decoder evaluates the same distribution $Q(\cdot\mid x_{<i})$, partitions its current interval in the same way, and determines which subinterval contains the encoded fraction. The corresponding symbol is $x_i$, which extends the prefix and determines the distribution used at the next position.

The model may change as the sequence is processed. It may condition on a fixed window, the complete prefix, or a state updated after every decoded symbol. Lossless decoding only requires the encoder and decoder to start from the same state and perform the same updates. They must also agree on where the sequence ends, either through a known length or a designated end symbol. If their probability distributions differ, their interval partitions differ and the bitstream no longer identifies the same sequence.

The claim made in the [ngrok article](https://ngrok.com/blog/compression-is-prediction) is exact at the level of this construction. Improving the predictor means increasing the probabilities assigned to the symbols that occur. This decreases

$$
\sum_{i=1}^n -\log_2Q(x_i\mid x_{<i})
$$

which is simultaneously the predictive log-loss and the ideal number of bits required by an entropy coder. This is also the equivalence used in [*Language Modeling Is Compression*](https://arxiv.org/abs/2309.10668), where language models provide conditional distributions and arithmetic coding turns their likelihoods into lossless representations.

The question posed by [Salvatore Sanfilippo](https://www.youtube.com/watch?v=UgRiVUce9sY) depends on what is meant by prediction. If prediction means selecting one continuation, compression and prediction are different tasks. A single guess cannot determine code lengths for all the other symbols that may occur. If prediction means assigning a conditional distribution to every possible next symbol, and its quality is measured by logarithmic loss, then the predictor and the compressor optimize the same quantity once an entropy coder connects probabilities to bits.

This equivalence assumes that the required conditional distributions are already available to both encoder and decoder. A real source does not normally reveal them. The compressor must estimate a model from previous data, transmit it, or construct it in a way that the decoder can reproduce. The bits spent because the model differs from the source have not yet been counted.


## The Source Is Unknown

The equivalence in the previous section begins with a model $Q$ that assigns a conditional distribution to every prefix. Its log-loss determines the ideal compressed length, and an entropy coder turns that length into a bitstream. An observed sequence, however, does not reveal the distribution that generated it. The encoder receives

$$
S=s_1,\ldots,s_n
$$

rather than the conditional probabilities of an unknown source.

We can make progress by restricting the distributions under consideration. The simplest restriction uses the same distribution at every position, independently of the preceding symbols. Let $q(a)$ be the probability assigned to symbol $a\in\Sigma$. The probability assigned to the complete sequence is then

$$
q(S)=\prod_{i=1}^n q(s_i)
$$

If $n_a$ denotes the number of occurrences of $a$ in $S$, equal factors can be collected to obtain

$$
q(S)=\prod_{a\in\Sigma}q(a)^{n_a}
$$

The corresponding log-loss is

$$
-\log_2q(S) = \sum_{a\in\Sigma}n_a\log_2\frac{1}{q(a)}
$$

Once the sequence has been observed, these counts determine which distribution in this restricted family assigns it the smallest loss. Define

$$
\widehat{P}_S(a)=\frac{n_a}{n}
$$

For any other distribution $q$,

$$
\begin{aligned}
-\log_2q(S) - \sum_{a\in\Sigma}n_a\log_2\frac{n}{n_a} &= \sum_{a\in\Sigma} n_a\log_2\frac{\widehat{P}_S(a)}{q(a)} \\
&\geq 0
\end{aligned}
$$

where the inequality is the same non-negativity argument used in the proof of the Source Coding Theorem. The empirical frequencies therefore minimize the log-loss among all models that use one fixed distribution throughout the sequence.

The resulting cost per symbol is the zero-order empirical entropy:

$$
\mathcal{H}_0(S) = \sum_{a\in\Sigma} \frac{n_a}{n} \log_2\frac{n}{n_a}
$$

Terms with $n_a=0$ contribute zero. Multiplying by $n$ gives

$$
n\mathcal{H}_0(S) = \sum_{a\in\Sigma} n_a\log_2\frac{n}{n_a} = -\log_2\widehat{P}_S(S)
$$

Unlike Shannon entropy, $\mathcal{H}_0(S)$ is not defined from a distribution that exists independently of the data. It is a property of the individual sequence $S$, obtained by fitting a zero-order model to its observed symbol frequencies. It makes no claim that the sequence was actually produced by independent draws from that model.

For a bitvector $B$ of length $n$ containing $m$ ones, the empirical probabilities are $m/n$ for $1$ and $(n-m)/n$ for $0$. Its total zero-order empirical entropy is

$$
n\mathcal{H}_0(B) = m\log_2\frac{n}{m} + (n-m)\log_2\frac{n}{n-m}
$$

The same quantity can be reached without fitting a probabilistic model. Suppose that the decoder already knows $n$ and $m$. The bitvector then belongs to the family

$$
\mathcal{B}_{n,m} = \left\{ B\in\{0,1\}^n : B\text{ contains exactly }m\text{ ones} \right\}
$$

A member of this family is determined by choosing the $m$ positions that contain a one, so

$$
|\mathcal{B}_{n,m}| = \binom{n}{m}
$$

The counting bound from the first section says that identifying an arbitrary member of this family requires

$$
\log_2\binom{n}{m}
$$

bits, up to integer rounding.

The relationship with empirical entropy can be established without relying on an asymptotic approximation. Set $p=m/n$ and consider the zero-order model that assigns probability $p$ to a one. Every member of $\mathcal{B}_{n,m}$ receives the same probability

$$
p^m(1-p)^{n-m} = 2^{-n\mathcal{H}_0(B)}
$$

The total probability assigned to the family is therefore

$$
\binom{n}{m}2^{-n\mathcal{H}_0(B)}
$$

Since this probability cannot exceed one,

$$
\log_2\binom{n}{m} \leq n\mathcal{H}_0(B)
$$

For the reverse bound, the number of ones produced by the model can take only the $n+1$ values from $0$ to $n$. When $p=m/n$, the value $m$ has maximum probability among these possible counts. Its probability must consequently be at least $1/(n+1)$. Hence

$$
\binom{n}{m}2^{-n\mathcal{H}_0(B)} \geq \frac{1}{n+1}
$$

Taking logarithms gives

$$
n\mathcal{H}_0(B)-\log_2(n+1) \leq \log_2\binom{n}{m} \leq n\mathcal{H}_0(B)
$$

Consequently,

$$
\log_2\binom{n}{m} = n\mathcal{H}_0(B)-O(\log n)
$$

This is the binary instance of the [method of types](https://www.researchgate.net/publication/220683915_The_Method_of_Types). The counting argument identifies the bitvector among all sequences with the same number of ones. The empirical model assigns each symbol its observed frequency and measures the log-loss of the resulting sequence. The two constructions differ by at most $\log_2(n+1)$ bits because the probabilistic model also distributes probability across sequences with other values of $m$, while the counting argument conditions on $m$ being known.

If $m$ is not known to the decoder, it must be represented as well. There are $n+1$ possible values, so transmitting it requires at most $\lceil\log_2(n+1)\rceil$ bits with a fixed-width representation. After this cost is included, the combinatorial and probabilistic descriptions agree within lower-order terms.

The same relation extends to a general alphabet. For a sequence $S$ with symbol counts $(n_a)_{a\in\Sigma}$, the number of sequences with the same composition is the multinomial coefficient

$$
\frac{n!}{\prod_{a\in\Sigma}n_a!}
$$

Its logarithm satisfies

$$
n\mathcal{H}_0(S)-O(|\Sigma|\log n) \leq \log_2\frac{n!}{\prod_{a\in\Sigma}n_a!} \leq n\mathcal{H}_0(S)
$$

The counting bound and the best zero-order log-loss therefore measure the same information up to the cost of describing the composition. The first counts the sequences that remain possible once the counts are fixed. The second fits a probability distribution from those counts and evaluates the observed sequence under that distribution.

Both depend only on the composition of $S$. Reordering its symbols leaves every $n_a$ unchanged and therefore leaves $\mathcal{H}_0(S)$ unchanged. A zero-order model assigns the same probability to a symbol wherever it occurs, even when the preceding symbols make some continuations more likely than others.

The sequential factorization from the previous section suggests how to remove this restriction. Instead of fitting one distribution to all positions, group positions according to their preceding context. Fix a context length $k$. For each string $\omega\in\Sigma^k$, let $n_{\omega a}$ be the number of positions at which the preceding $k$ symbols are $\omega$ and the current symbol is $a$. Write

$$
n_\omega = \sum_{a\in\Sigma}n_{\omega a}
$$

for the number of symbols observed after context $\omega$.

Within that context, the empirical conditional distribution is

$$
\widehat{P}_S(a\mid\omega) = \frac{n_{\omega a}}{n_\omega}
$$

Using this distribution whenever the preceding context is $\omega$ assigns the symbols following $\omega$ a total log-loss of

$$
\sum_{a\in\Sigma} n_{\omega a} \log_2\frac{n_\omega}{n_{\omega a}}
$$

Summing over all contexts gives

$$
n\mathcal{H}_k(S) = \sum_{\omega\in\Sigma^k} \sum_{a\in\Sigma} n_{\omega a} \log_2\frac{n_\omega}{n_{\omega a}}
$$

This defines the [$k$-th order empirical entropy](https://arxiv.org/pdf/0708.2084). Equivalently, let $S_\omega$ be the sequence formed by collecting the symbols that follow occurrences of $\omega$. Then

$$
\mathcal{H}_k(S) = \frac{1}{n} \sum_{\omega\in\Sigma^k} |S_\omega|\mathcal{H}_0(S_\omega)
$$

The first $k$ symbols have no complete length-$k$ context. They may be encoded separately, which changes the total cost by at most $k\log_2|\Sigma|$ bits, or handled through an agreed boundary convention.

When $k=0$, there is only the empty context, and the definition reduces to $\mathcal{H}_0(S)$. For $k>0$, different contexts receive different empirical distributions. The quantity $n\mathcal{H}_k(S)$ is the smallest log-loss obtained on $S$ by assigning one distribution to each observed length-$k$ context. It is therefore the empirical counterpart of the conditional description length from the previous section.

Increasing $k$ gives the model more information about each position. It can distinguish occurrences that a shorter context would place in the same group, and its fitted log-loss cannot increase. For sufficiently large $k$, most contexts may occur only once and determine their following symbol perfectly. The resulting value of $\mathcal{H}_k(S)$ may then approach zero.

The sequence has not become describable with zero bits. The empirical distributions were constructed after inspecting the sequence, while the decoder has not yet seen it. The symbol counts required by $\mathcal{H}_0(S)$ and the context tables required by $\mathcal{H}_k(S)$ must either be transmitted, learned through a procedure reproduced by the decoder, or already shared. Empirical entropy measures the cost of the data under the fitted model, but it does not include the cost of making that model available.


## The Model Is Part of the Message

Increasing the context length gives the model more information about each symbol. For a fixed sequence $S$, sufficiently long contexts occur rarely, and some occur only once. Whenever an observed context $\omega$ has a single observed continuation, its empirical distribution assigns probability one to that continuation. The corresponding contribution to the code length is $-\log_2 1=0$. If every relevant context determines its continuation in this way, the empirical entropy can fall all the way to

$$
\mathcal{H}_k(S)=0
$$

The sequence has not become free to describe. The probabilities used in this calculation were obtained from the sequence itself. A decoder can assign probability one to the correct continuation only if it knows which continuation followed each context. For an alphabet of size $\sigma$, a direct table containing one distribution for every context in $\Sigma^k$ may require on the order of

$$
\sigma^{k+1}\log_2 n
$$

bits to store its counts. Sparse representations can avoid entries for contexts that never occur, but they must still identify the observed contexts and their continuations. As $k$ grows, the encoded sequence may become shorter because information has moved into the model that describes it.

Let $M$ contain everything the decoder needs to reproduce the probabilities used by the encoder. The length of a complete description is then

$$
L(M,S)=L(M)+L(S\mid M)
$$

where $L(M)$ describes the model and $L(S\mid M)$ describes the sequence using that model. Minimizing only the second term rewards any model capable of remembering the input. Minimizing their sum rewards a larger model only when the reduction in the second term pays for its description. This two-part accounting is the simplest form of the [minimum description length principle](https://web.mit.edu/6.433/www/handouts/minimumdescriptionlength.pdf).

The same accounting separates three quantities that are easily conflated when prediction is described as compression. Suppose that objects are generated according to a source distribution $P$, while the compressor assigns probabilities according to a model $Q$. The entropy of the source is

$$
H(P) = -\sum_x P(x)\log_2 P(x)
$$

If the true distribution were available, this would be the smallest achievable expected description length, apart from the integer rounding and termination costs of a concrete code. The compressor does not usually know $P$. When it uses $Q$, its expected ideal code length becomes the [cross-entropy](https://arxiv.org/html/2309.10668v2#S2.SS6)

$$
H(P,Q) = -\sum_x P(x)\log_2 Q(x)
$$

Here $Q(x)$ must be positive whenever $P(x)$ is positive. Otherwise an object that can be produced by the source receives zero probability from the model and requires an unbounded ideal code length.

Subtracting the source entropy from the cross-entropy gives

$$
\begin{aligned}
H(P,Q)-H(P) &= \sum_x P(x)\log_2\frac{P(x)}{Q(x)} \\
&= D_{\mathrm{KL}}(P\mathbin\Vert Q)
\end{aligned}
$$

The quantity $D_{\mathrm{KL}}(P\mathbin\Vert Q)$ is the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) from the source to the model. It is non-negative and vanishes when $P$ and $Q$ agree on the outcomes that the source can produce. Therefore,

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\mathbin\Vert Q)
$$

For sequences, $P$ and $Q$ can be read as distributions over complete strings. Factoring both distributions into conditional probabilities also factors their divergence:

$$
D_{\mathrm{KL}}(P_{1:n}\mathbin\Vert Q_{1:n}) = \sum_{i=1}^{n} \mathbb{E}_{X_{<i}\sim P} \left[ D_{\mathrm{KL}} \left( P(\cdot\mid X_{<i}) \mathbin\Vert Q(\cdot\mid X_{<i}) \right) \right]
$$

Each term measures the expected number of extra bits paid at one position because the predictive distribution differs from the source distribution. Better predictions reduce these mismatch costs, and an entropy coder converts that reduction into a shorter bitstream.

The [ngrok article](https://ngrok.com/blog/compression-is-prediction) is right that better probability estimates lead to better compression. The quantity reduced by a better model is $H(P,Q)$, through a reduction of $D_{\mathrm{KL}}(P\mathbin\Vert Q)$. For a fixed source $P$, improving $Q$ does not change $H(P)$. Calling both quantities entropy hides the difference between uncertainty produced by the source and extra bits paid because the compressor uses the wrong probabilities.

Empirical entropy adds one more qualification. The distribution used to compute $\mathcal{H}_k(S)$ is fitted to the same sequence whose length is being measured. Increasing $k$ can reduce this fitted cost even when the total description becomes longer. The relevant comparison is therefore not between models according to $\mathcal{H}_k(S)$ alone, but between their complete lengths:

$$
\min_M \bigl(L(M)+L(S\mid M)\bigr)
$$

Whether $L(M)$ appears in the transmitted bitstream depends on what the encoder and decoder already share. If the model is fixed by the file format, the protocol, or an external agreement, it need not be sent with every sequence. Its cost has become shared information and may be amortized across many messages, but it has not ceased to exist.

If the model is fitted before compression and is not already available to the decoder, its tables, parameters, or weights must accompany the encoded data. The paper [*Language Modeling Is Compression*](https://arxiv.org/html/2309.10668v2#S3.SS2) calls the size obtained without those parameters the *raw compression rate*. Its *adjusted compression rate* adds the model size to the encoded data. A larger language model may obtain a lower log-loss while producing a worse adjusted rate when its parameters are used to compress too little data.

A third possibility is to let encoder and decoder learn the model in the same order. They begin from the same state, encode a symbol using the current distribution, update the model after that symbol becomes known, and repeat. The decoder can reconstruct every update from the prefix it has already decoded, so the final model does not need to be transmitted. This [prequential or online code](https://arxiv.org/html/2309.10668v2#S3.SS2) replaces a separate parameter description with the extra bits paid while the model is still learning. The update rule, its initial state, and any randomness affecting it must still be shared.

Once these costs are included, compression measures more than the quality of a predictor. It measures how concisely the predictor and its remaining errors describe the data together. Even that complete length only concerns reconstruction of the whole sequence. A representation may occupy few bits while still requiring the entire sequence to be decoded before any part of it can be used.


## The Bitstream Is Not Always the Final Object

The length $L(M)+L(S\mid M)$ measures the number of bits needed to describe the model and reconstruct the sequence. It does not determine what can be done with those bits before the reconstruction is complete.

Consider a vector

$$
A=(a_0,\ldots,a_{n-1})
$$

whose values belong to $\{0,\ldots,u-1\}$. Let

$$
b=\lceil\log_2 u\rceil
$$

A [bit-packed representation](https://lukefleed.xyz/posts/compressed-fixedvec/) assigns exactly $b$ consecutive bits to each value, using $nb$ bits for the payload, apart from alignment and metadata. The position of $a_i$ begins at bit $ib$, so the representation can recover it by reading the machine words that intersect those $b$ bits, shifting them, and applying a mask. When $b$ fits within a machine word, this takes a constant number of memory accesses and arithmetic operations.

This representation does not exploit differences in frequency. Every value receives the same number of bits. If the values follow a non-uniform distribution, or if their probabilities depend on earlier values, an entropy coder may produce a shorter stream:

$$
L(A\mid M) \approx \sum_{i=0}^{n-1} -\log_2 Q(a_i\mid a_{<i})
$$

The shorter stream provides a different access contract. In an ordinary arithmetic-coded stream, the decoding state at position $i$ depends on the symbols that precede it. If the model also uses their context, its next distribution depends on the same prefix. Recovering $a_i$ therefore requires decoding from the beginning of the stream or from an earlier checkpoint whose coding and model states have been stored.

The bit-packed vector may occupy more space while answering `vector[i]` directly. Its compression comes from restricting the possible value at each position to an alphabet of size $u$, rather than from predicting which value will occur. When all values remain equally plausible, fixed-width packing uses the information supplied by that restriction without inventing a non-uniform model.

A compressed archive needs an encoding $E$ and a decoder $D$ satisfying

$$
D(E(A))=A
$$

Once the whole vector can be reconstructed, the encoding has fulfilled its contract. A compressed representation may be required to satisfy a stronger condition. It must support an access algorithm such that

$$
\operatorname{Access}(R(A),i)=a_i
$$

without first reconstructing all of $A$.

Checkpoints can give an entropy-coded stream faster access, but each checkpoint occupies space. Smaller blocks reduce the amount of decoding required for an access and increase the number of stored states. Larger blocks save metadata and increase access time. The objective is no longer to minimize the bitstream without qualification. It becomes

$$
\min_R |R(A)| \qquad \text{subject to} \qquad T_{\operatorname{Access}}(R)\leq t
$$

for a chosen access-time bound $t$.

The operation need not be random access, and the object need not be a vector. The same issue arises whenever compressed data must be searched, traversed, compared, or partially decoded. The required operations constrain which short descriptions are useful and how much auxiliary information they need.

An entropy coder answers how few bits are required to reproduce an object under a probability model. A compressed representation must also encode enough structure to perform the operations expected by its users. The decoder’s task is therefore part of the compression problem, together with the family of possible objects and the information already shared with it.

## So, Is Compression Prediction?

For a probabilistic sequence, the connection can now be stated precisely. The probability assigned to a complete sequence factors as

$$
P(x_{1:n}) = \prod_{i=1}^{n}P(x_i\mid x_{<i})
$$

and its ideal description length is

$$
-\log_2 P(x_{1:n}) = \sum_{i=1}^{n} -\log_2 P(x_i\mid x_{<i})
$$

A predictor that returns each conditional distribution supplies the probabilities needed by an entropy coder. Its cumulative log-loss is the ideal length of the resulting encoding. Improving those distributions reduces cross-entropy by reducing the divergence between the model and the source. Arithmetic coding converts the resulting probabilities into a lossless bitstream whose length approaches that cumulative loss.

Within these conditions, prediction and compression optimize the same quantity. Prediction must mean assigning a probability distribution to every possible continuation, rather than selecting a single continuation. Performance must be measured by log-loss. Encoder and decoder must reproduce the same probabilities. The probabilities must then be converted into a decodable code.

Compression begins before any of these conditions are imposed. If an object belongs to a finite family $\mathcal{F}$, distinguishing its members requires about $\log_2|\mathcal{F}|$ bits even when no probability distribution is available. A source distribution refines this count by making some alternatives cheaper than others. When the object is a sequence, the chain rule rewrites those costs as successive conditional predictions. Prediction enters because the probability of a sequence admits this decomposition, not because every compressed object was originally a prediction problem.

The resulting code length still omits information that the decoder may not possess. A fitted model must be shared, transmitted, or reconstructed online. A short sequential stream may also omit the structure required for direct access. Once these costs are included, a compressor is determined not only by its predictor, but by the complete agreement between encoder and decoder.

The phrase *compression is prediction* identifies an exact and useful correspondence inside this agreement. It explains why language-model log-loss can be interpreted in bits and why a better probabilistic model can produce a shorter arithmetic code. Used as a definition of compression, however, it leaves out the choice of possible objects, the cost of specifying the model, and the operations that the encoded representation must continue to support.

> Compression is prediction in the precise setting of probabilistic sequential modeling under log-loss. More generally, prediction is one way to assign description lengths. Compression asks for the shortest representation relative to the objects considered, the information shared with the decoder, and the operations the representation must support.
