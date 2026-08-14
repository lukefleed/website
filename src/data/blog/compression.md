---
author: Luca Lombardo
pubDatetime: 2026-08-14T00:00:00Z
title: Is Compression Really Prediction?
slug: compression
featured: true
draft: false
description: Which part of compression is prediction, and what must already be fixed before that equivalence describes a complete compressor?
---

Over the past few weeks, I have repeatedly encountered the same claim on Hacker News: **compression is prediction**. An [ngrok article](https://ngrok.com/blog/compression-is-prediction) develops the idea through entropy coding and language models, while [Salvatore Sanfilippo](https://www.youtube.com/watch?v=UgRiVUce9sY) recently discussed the similarities and differences between predicting and compressing. Both start from a real mathematical connection. A model assigns probabilities to possible continuations, and an entropy coder turns those probabilities into shorter or longer representations.

I have spent the last few years working on compression, information theory, and compressed representations. I agree with the mathematics behind the claim, but as a general account of compression it seems to begin halfway through the story. Before assigning probabilities or predicting the next symbol, we must decide what the object is, which alternatives the decoder already knows to be possible, and what information the representation must preserve.

Compression can be defined before introducing a probability distribution. If an object belongs to a finite family, then representing it means distinguishing it from every other member of that family. The cardinality of the family already determines a lower bound on the number of bits required. If we consider an individual object instead, its shortest effective description leads to Kolmogorov complexity. Neither viewpoint requires a notion of the next symbol.

Probability becomes relevant when the possible objects should not all receive descriptions of the same length. A distribution tells us which objects may be assigned shorter descriptions and which ones must receive longer descriptions in exchange. When an object is generated sequentially, the probability of the whole object can then be factored into conditional probabilities for its successive symbols. This is the point at which lossless compression and prediction meet.

The question is therefore not whether the slogan can be proved under a probabilistic model. It is which part of compression that proof captures, what the encoder and decoder must already share, and which requirements remain outside the resulting bit count.

## Table of Contents

## Compression Before Probability

Compression begins with descriptions. Given an object $x$, we seek a binary string from which a decoder can recover $x$ without ambiguity. A representation is shorter when it identifies the same object using fewer bits, but its length is meaningful only relative to what the encoder and decoder have already agreed upon. The description language, the class of admissible objects, and any information available to both sides determine what still needs to be encoded.

[Kolmogorov complexity](https://link.springer.com/book/10.1007/978-3-030-11298-1) captures the most general version of this idea. After fixing a universal machine $U$, a program for $U$ acts as a description, and the Kolmogorov complexity of a binary string $x$ is

$$
K_U(x) = \min \{ |p| : U(p) = x \}
$$

Thus, $K_U(x)$ is the length of the shortest program that produces $x$. Any regularity that can be expressed algorithmically may contribute to a shorter description, independently of whether that regularity was expressed as a probability distribution. The definition concerns the description of a complete individual object, not the prediction of one of its symbols from the preceding ones.

The choice of $U$ affects the value of $K_U(x)$. The invariance theorem limits this dependence: changing from one universal machine to another changes the complexity by at most an additive constant independent of $x$. This result does not provide a compression algorithm. Kolmogorov complexity is not computable, so no procedure can determine the shortest program for every string, much less construct it.

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

Conversely, the members of $\mathcal{F}$ can be indexed and their indices represented in binary, so the fixed-length bound can be attained. The quantity

$$
\log_2 |\mathcal{F}|
$$

is the *counting bound* of the family. It measures the information required to identify an arbitrary member of $\mathcal{F}$ when no member may remain indistinguishable from another. In some succinct data-structure literature, the same quantity is called the *worst-case entropy* of the family. I will use *counting bound* because its derivation does not assume that objects are sampled uniformly, or that they are sampled at all. It only counts the admissible objects and the distinct binary representations available to identify them.

The family $\mathcal{F}$ is part of the problem specification. If the decoder knows only that the object belongs to a larger family $\mathcal{G}$, then the representation must distinguish among the members of $\mathcal{G}$, and the corresponding bound becomes $\log_2 |\mathcal{G}|$. A smaller description is possible only when the restriction from $\mathcal{G}$ to $\mathcal{F}$ is already known or is itself communicated to the decoder. What counts as redundancy therefore depends on which alternatives the representation is required to distinguish.

The counting bound treats all members of $\mathcal{F}$ symmetrically. It determines the length required when every object must fit within the same bound, but it cannot express that some objects occur more frequently than others. To exploit such an imbalance, a code must assign different lengths to different objects. Determining which descriptions should be shorter requires a probability distribution, and this changes the objective from worst-case length to expected length.

## Possibilities Have Different Probabilities

To express these frequencies, we replace the family $\mathcal{F}$ with a source whose possible outputs form a finite set $\mathcal{X}$. For each $x\in\mathcal{X}$, the source specifies a probability

$$
P(x)=\Pr(X=x)
$$

where $P(x)>0$ and

$$
\sum_{x\in\mathcal{X}}P(x)=1
$$

A probability must now be translated into a quantity measured in bits. Since the probability of two independent outcomes is the product of their individual probabilities, the corresponding bit costs should add. The logarithm performs exactly this conversion. The [information content](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) of an outcome $x$ is

$$
I_P(x)=-\log_2P(x)
$$

More probable outcomes have smaller information content. If an event has probability $2^{-b}$, observing it provides $b$ bits of information according to this definition.

Before the source produces an outcome, its information content is not known. Its average value is

$$
H(P) = \sum_{x\in\mathcal{X}}P(x)I_P(x) = -\sum_{x\in\mathcal{X}}P(x)\log_2P(x)
$$

This quantity is the [Shannon entropy](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) of the source. It depends on the distribution $P$, rather than on the particular names assigned to the objects in $\mathcal{X}$. When $P$ is uniform, every object has information content $\log_2|\mathcal{X}|$, so Shannon entropy equals the counting bound for the uniform family.

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

A source distribution may assign one probability to a complete object. Once that object is an ordered sequence $x_{1:n}=x_1,\ldots,x_n$, the same probability admits a second description: one factor for each symbol, conditioned on the prefix that precedes it.

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

During generation, a model may choose or sample a symbol from its next-symbol distribution. During compression, the encoder already knows the actual next symbol and uses the probability assigned to it. Prediction here means constructing the distribution, not replacing the next symbol with a guess.

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

The claim made in the [ngrok article](https://ngrok.com/blog/compression-is-prediction) is exact at the level of this construction. For a fixed observed sequence, a predictor improves under logarithmic loss when it reduces $\mathcal{L}_Q(x_{1:n})$, equivalently when it increases $Q(x_{1:n})$. That same loss is the ideal length of the data term produced by an entropy coder. This is also the equivalence used in [*Language Modeling Is Compression*](https://arxiv.org/html/2309.10668v2), where language models provide conditional distributions and arithmetic coding turns their likelihoods into lossless representations.

This equivalence assumes that the required conditional distributions are available to both encoder and decoder. A real source does not normally reveal them. The compressor must estimate a model from previous data, transmit it, or construct it in a way that the decoder can reproduce. The cost of making the model available has not been included, and the excess length caused by a mismatch between the model and the source has not yet been separated from the source’s own uncertainty.


## The Source Is Unknown

The encoder observes

$$
S=s_1,\ldots,s_n
$$

rather than the conditional probabilities of the source that produced it. A data-dependent measure must therefore begin with a model family that can be fitted from $S$. The simplest family uses the same distribution at every position, independently of the preceding symbols. Let $q(a)$ be the probability assigned to symbol $a\in\Sigma$. The probability assigned to the complete sequence is then

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

The resulting cost per symbol is the [zero-order empirical entropy](https://arxiv.org/abs/0708.2084):

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

The counting bound says that identifying an arbitrary member of this family requires

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

This is the binary instance of the [method of types](https://ieeexplore.ieee.org/document/720546/). The counting argument identifies the bitvector among all sequences with the same number of ones. The empirical model assigns each symbol its observed frequency and measures the log-loss of the resulting sequence. The two constructions differ by at most $\log_2(n+1)$ bits because the probabilistic model also distributes probability across sequences with other values of $m$, while the counting argument conditions on $m$ being known.

If $n$ is known but $m$ is not, then $m$ must be represented as well. There are $n+1$ possible values, so transmitting it requires at most $\lceil\log_2(n+1)\rceil$ bits with a fixed-width representation. After this cost is included, the combinatorial and probabilistic descriptions agree within lower-order terms.

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

Zero-order models cannot use the order that made sequential prediction useful. To introduce context without assuming a known source, group positions according to their preceding context. Fix a context length $k$. For each string $\omega\in\Sigma^k$, let $n_{\omega a}$ be the number of positions at which the preceding $k$ symbols are $\omega$ and the current symbol is $a$. Write

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

This defines the [$k$-th order empirical entropy](https://arxiv.org/abs/0708.2084). Equivalently, let $S_\omega$ be the sequence formed by collecting the symbols that follow occurrences of $\omega$. Then

$$
\mathcal{H}_k(S) = \frac{1}{n} \sum_{\omega\in\Sigma^k} |S_\omega|\mathcal{H}_0(S_\omega)
$$

The first $k$ symbols have no complete length-$k$ context. They may be encoded together in $\lceil k\log_2|\Sigma|\rceil$ bits, or handled through an agreed boundary convention.

When $k=0$, there is only the empty context, and the definition reduces to $\mathcal{H}_0(S)$. For $k>0$, different contexts receive different empirical distributions. The quantity $n\mathcal{H}_k(S)$ is the smallest log-loss on the positions with a complete length-$k$ context, obtained by assigning one distribution to each observed context. Together with the separately encoded prefix, it is the empirical counterpart of sequential conditional log-loss.

Longer contexts refine these groups, so their fitted log-loss cannot increase on the positions covered by both models. The first $k$ symbols remain governed by the boundary convention stated above. For sufficiently large $k$, most observed contexts occur only once and determine their following symbol. The empirical term can then collapse to

$$
\mathcal{H}_k(S)=0
$$


## The Model Is Part of the Message

The value $\mathcal{H}_k(S)=0$ measures only the data term after the fitted context model is available. A decoder can assign probability one to a continuation only if it knows which continuation followed that context. For an alphabet of size $\sigma$, a direct table has $\sigma^k$ contexts and $\sigma$ counts per context. Representing each count in $O(\log_2(n+1))$ bits requires

$$
O\!\left(\sigma^{k+1}\log_2(n+1)\right)
$$

bits. A sparse representation avoids entries for contexts that never occur, but it must still identify the observed contexts and their continuations. The data term becomes shorter because information has moved into the model.

Let $M$ contain everything the decoder needs to reproduce the probabilities used by the encoder. A complete two-part description has length

$$
L(M,S)=L(M)+L(S\mid M)
$$

where $L(M)$ describes the model and $L(S\mid M)$ describes the sequence using that model. Minimizing only the second term rewards a model that memorizes the input. A larger model improves the complete description only when the reduction in $L(S\mid M)$ pays for its own description. This two-part accounting is the simplest form of the [minimum description length principle](https://doi.org/10.7551/mitpress/1114.003.0005).

The data term $L(S\mid M)$ contains a second distinction. Suppose that objects are generated according to a source distribution $P$, while the compressor assigns probabilities according to a model $Q$. The source entropy $H(P)$, defined earlier, is the lower bound on the expected description length when the true distribution is available. When the compressor uses $Q$, its expected ideal data length becomes the [cross-entropy](https://arxiv.org/html/2309.10668v2#S2.SS6)

$$
H(P,Q) = -\sum_x P(x)\log_2 Q(x)
$$

We retain the support condition required for lossless coding: $Q(x)>0$ whenever $P(x)>0$.

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

For sequences, $P$ and $Q$ can be read as distributions over complete strings. Factoring both distributions into conditional probabilities gives the [chain rule for relative entropy](https://sites.stat.columbia.edu/liam/teaching/neurostat-spr11/papers/EM/Cover%26Thomas-Ch2.pdf):

$$
D_{\mathrm{KL}}(P_{1:n}\mathbin\Vert Q_{1:n}) = \sum_{i=1}^{n} \mathbb{E}_{X_{<i}\sim P} \left[ D_{\mathrm{KL}} \left( P(\cdot\mid X_{<i}) \mathbin\Vert Q(\cdot\mid X_{<i}) \right) \right]
$$

Each term measures the expected number of extra bits paid at one position because the predictive distribution differs from the source distribution. Better predictions reduce these mismatch costs, and an entropy coder converts that reduction into a shorter bitstream.

The [ngrok article](https://ngrok.com/blog/compression-is-prediction) correctly links better probability estimates to shorter codes, and it later acknowledges model size and computation as practical overheads. The narrower correction concerns terminology. For a fixed source $P$, improving $Q$ reduces $D_{\mathrm{KL}}(P\mathbin\Vert Q)$ and therefore $H(P,Q)$. It does not reduce $H(P)$. Calling cross-entropy simply entropy conflates uncertainty produced by the source with extra bits paid because the compressor uses the wrong probabilities.

Whether $L(M)$ appears in the transmitted bitstream depends on what the encoder and decoder already share. If the model is fixed by the file format, the protocol, or an external agreement, it need not be sent with every sequence. Its cost has become shared information and may be amortized across many messages, but it has not ceased to exist.

If the model is fitted before compression and is not already available to the decoder, its tables, parameters, or weights must accompany the encoded data. [*Language Modeling Is Compression*](https://arxiv.org/html/2309.10668v2#S3.SS2) calls the ratio computed without parameter size the *raw compression rate*. The *adjusted compression rate* includes the parameter size in the compressed size. A larger language model may obtain a lower log-loss while producing a worse adjusted rate when its parameters are amortized over too little data.

A third possibility is to let encoder and decoder learn the model in the same order. They begin from the same state, encode a symbol using the current distribution, update the model after that symbol becomes known, and repeat. The decoder reconstructs every update from the prefix it has already decoded, so the final parameters need not be transmitted. A [prequential or online code](https://arxiv.org/html/2309.10668v2#S3.SS2) instead includes the training procedure and pays additional log-loss while the model is learning. The initialization, update rule, training program, and any randomness affecting them must remain shared.


## The Bitstream Is Not Always the Final Object

Every code considered so far has been judged by one operation: reconstructing the complete object. The two-part description length says nothing about what can be done with the encoded data before that reconstruction is complete.

Consider a vector

$$
A=(a_0,\ldots,a_{n-1})
$$

whose values belong to $\{0,\ldots,u-1\}$. Let

$$
b=\lceil\log_2 u\rceil
$$

A [bit-packed representation](https://lukefleed.xyz/posts/compressed-fixedvec/) assigns exactly $b$ consecutive bits to each value, using $nb$ bits for the payload, apart from alignment and metadata. The position of $a_i$ begins at bit $ib$. If a storage word contains at least $b$ bits, recovering $a_i$ requires reading at most two adjacent words, shifting them, and applying a mask. The addresses and shifts are computed directly from $i$, so access takes $O(1)$ time.

This representation does not exploit differences in frequency. Every value receives the same number of bits. If the values follow a non-uniform distribution, or if their probabilities depend on earlier values, an entropy coder may produce a shorter stream:

$$
L(A\mid M) \approx \sum_{i=0}^{n-1} -\log_2 Q(a_i\mid a_{<i})
$$

The shorter stream provides a different access contract. In an ordinary arithmetic-coded stream, the decoding state at position $i$ depends on the symbols that precede it. If the model also uses their context, its next distribution depends on the same prefix. Recovering $a_i$ therefore requires decoding from the beginning of the stream or from an earlier checkpoint whose coding and model states have been stored.

The bit-packed vector may occupy more space while answering `vector[i]` directly. Its compression comes from restricting the possible value at each position to an alphabet of size $u$, rather than from predicting which value will occur. When all values remain equally plausible, fixed-width packing uses the information supplied by that restriction without requiring a non-uniform model.

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

The bitstream length answers the archive problem. A compressed representation must also encode enough structure for its required operations. The decoder contract now has three explicit parts: the objects it must distinguish, the information it already shares with the encoder, and the operations it must perform without full reconstruction. Prediction determines conditional code lengths inside this contract. It does not determine the contract itself.

## So, Is Compression Prediction?

[Salvatore Sanfilippo’s](https://youtu.be/UgRiVUce9sY) question depends on what prediction means. Selecting one continuation is not enough to compress a sequence. Assigning a conditional distribution to every possible continuation is enough to determine its ideal code lengths. The [ngrok article](https://ngrok.com/blog/compression-is-prediction) adopts this second meaning and correctly identifies cumulative log-loss with the ideal length of an entropy-coded payload. [*Language Modeling Is Compression*](https://arxiv.org/html/2309.10668v2) uses the same identity and extends the accounting to model parameters through raw, adjusted, and prequential compression rates. 

What I felt they do not capture is the complete compression problem surrounding that identity. Before a predictor can assign probabilities, encoder and decoder must agree on which objects the representation must distinguish. This choice already determines a counting bound, even without a probabilistic model. The model must then be shared, reconstructed from the decoded prefix, or included in the description. Finally, reconstructing the complete sequence may not be the only required operation. Random access, search, and partial decoding impose constraints that predictive loss does not express.

These omitted terms can change which representation is best. A model with lower log-loss can produce a larger file once its parameters are included. An arithmetic-coded stream can use fewer bits than a bit-packed vector while failing to provide constant-time access. Even a vanishing empirical entropy does not produce a zero-length representation when the decoder still needs the fitted model. A shorter predictive data term is therefore neither a guarantee of a shorter complete description nor a guarantee of a more useful compressed representation.

Compression is prediction exactly for the data term induced by a shared sequential probability model under logarithmic loss. It describes the whole compressor only when the admissible objects and decoder contract have already been fixed, the model introduces no unaccounted cost, and complete reconstruction is the only required operation. Otherwise, prediction determines one term of the compression objective. It does not determine the objective itself.

