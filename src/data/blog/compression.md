---
author: Luca Lombardo
pubDatetime: 2026-08-15T00:00:00Z
title: Is Compression Really Prediction?
slug: compression
featured: false
draft: false
description: When does lossless compression reduce to prediction, and what must already be fixed before that equivalence applies?
---

Over the past few weeks, I have repeatedly encountered the same claim on Hacker News: **compression is prediction**. The recent discussion has approached it from both directions. Two 3Blue1Brown videos, [_Reinventing Entropy_](https://youtu.be/l6DKRf-fAAM) and [_But what is cross-entropy?_](https://youtu.be/GlYgs6v2YfU), derive entropy and cross-entropy from the limits of source coding. An [ngrok article](https://ngrok.com/blog/compression-is-prediction) follows the same mathematics through arithmetic coding and language models. [Salvatore Sanfilippo](https://www.youtube.com/watch?v=UgRiVUce9sY) asks how far the resulting identification between prediction and compression should be taken.

These explanations meet at one fact. A probabilistic model assigns a conditional probability to every possible continuation, and an entropy coder converts the probability assigned to the observed continuation into bits. For a sequence $x_{1:n}$ and a model $Q$, the resulting ideal payload length is

$$
-\log_2 Q(x_{1:n}) = \sum_{i=1}^n -\log_2 Q(x_i\mid x_{<i})
$$

up to the overhead introduced by the coding procedure. The quantity on the right is also the model’s cumulative logarithmic loss. In this setting, improving prediction under log-loss and reducing the encoded payload are the same optimization problem.

I have spent the last few years working on compression, information theory, and compressed data structures and wanted to give my two cents. I agree with this equivalence, but I do not think it describes the complete compression problem. It applies after several choices have already been made. The encoder and decoder must agree on what kind of object is being represented, which alternatives remain possible, how the probability model is made available, and what the decoder must be able to do with the representation.

Throughout this article, _compression_ means lossless compression unless stated otherwise. Even within that scope, compression can be defined before introducing a sequential model. A finite family of admissible objects gives a counting lower bound without identifying a next symbol. A fixed or data-dependent code can later be interpreted probabilistically, and a distribution over serialized objects can be factored into next-symbol conditionals. That reinterpretation does not choose the family of objects, pay for information unavailable to the decoder, or enforce operations such as random access.

The question is therefore not whether prediction and compression can be made mathematically equivalent. They can. The question is what must be fixed before the equivalence applies, which part of a complete representation its bit count measures, and what remains outside that measurement.

> **A note on level.** This article is a bit technical, it assumes familiarity with undergraduate mathematics and elementary proof-style arguments, but no prior background in information theory is really required, although it may help.

## Table of Contents

## Compression Before Probability

The [ngrok article](https://ngrok.com/blog/compression-is-prediction) begins by distinguishing minification from what it calls “true” compression. A minifier removes comments, whitespace, and other parts of a source file that do not affect its execution. The resulting program is shorter, but the original source file cannot be reconstructed from it.

Whether this operation is lossless depends on what the representation is required to preserve. If the object is the original sequence of source bytes, minification is lossy. If the object is the program’s behaviour and the decoder may return any behaviourally equivalent program, a semantics-preserving minifier is lossless relative to that different contract. The transformation has not changed. The object being represented has.

This distinction precedes any probability model. Before asking how likely an object is, the encoder and decoder must agree on what counts as that object and when two decoded outputs count as equivalent. Only then does the length of a description become meaningful.

Once an individual object $x$ has been fixed, the most permissive effective descriptions are programs that produce it. After choosing a universal machine $U$, the [Kolmogorov complexity](https://en.wikipedia.org/wiki/Kolmogorov_complexity) of a binary string $x$ is

$$
K_U(x) = \min \left\{ |p| : U(p)=x \right\}
$$

Thus, $K_U(x)$ is the length of the shortest program that outputs $x$. Any regularity that can be expressed algorithmically may shorten this description. A string containing a billion zeros has a long literal representation but a short program that prints one billion zeros. The definition does not require the string to have been sampled from a source, and it does not require one symbol to be predicted from the symbols preceding it.

The machine $U$ is part of the description language. Choosing a different universal machine changes which programs are available and therefore changes the exact value of the complexity. The invariance theorem bounds this dependence. For two fixed universal machines $U$ and $V$, there is a constant $c_{U,V}$ such that

$$
\left| K_U(x)-K_V(x) \right| \leq c_{U,V}
$$

for every string $x$. The constant may depend on the two machines, but not on $x$. It accounts for the fixed program needed to simulate one description language in the other.

Kolmogorov complexity gives a limit on the effective description of an individual object, but it does not provide a general compression algorithm. The function $K_U$ is not computable. No procedure can determine the length of the shortest program for every string, much less construct that program. A practical compressor must restrict the descriptions it is willing and able to consider.

One such restriction is that the object belongs to a finite family $\mathcal{F}$. Once $\mathcal{F}$ has been fixed, a lossless representation must distinguish every member of that family from every other member. Consider a fixed-length encoding

$$
C : \mathcal{F} \longrightarrow \{0,1\}^{\ell}
$$

Lossless decoding requires $C$ to be injective. Since only $2^\ell$ binary strings of length $\ell$ exist, injectivity implies

$$
2^\ell \geq |\mathcal{F}|
$$

and therefore

$$
\ell \geq \left\lceil \log_2|\mathcal{F}| \right\rceil
$$

An agreed enumeration of $\mathcal{F}$ attains this bound by assigning each object an index and representing that index in binary. The quantity

$$
\log_2|\mathcal{F}|
$$

is the _counting bound_ of the family. In the literature on [succinct data structures](https://en.wikipedia.org/wiki/Succinct_data_structure), which studies representations whose space approaches information-theoretic lower bounds, the quantity $\log_2|\mathcal{F}|$ is sometimes called the _worst-case entropy_ of the family. I will use _counting bound_ because its derivation assumes neither a uniform distribution nor any sampling process. It only counts the alternatives that the representation must distinguish.

The family $\mathcal{F}$ is part of the information shared by the encoder and decoder. If the decoder knows only that the object belongs to a larger family $\mathcal{G}$, then the representation must distinguish among the members of $\mathcal{G}$ instead. The lower bound becomes

$$
\log_2|\mathcal{G}|
$$

A restriction from $\mathcal{G}$ to $\mathcal{F}$ saves bits only if the decoder already knows that restriction or if the representation communicates it. What counts as redundancy therefore depends on which alternatives have already been excluded.

Kolmogorov complexity and the counting bound answer different versions of the same preliminary question. The first considers the shortest effective description of one object. The second considers the number of bits needed to distinguish every object in a fixed finite family. Neither requires a probability distribution or a next-symbol predictor.

The counting bound treats all members of $\mathcal{F}$ symmetrically. It determines the optimal worst-case length when every object must fit within the same number of bits. It cannot express that some objects should receive shorter descriptions because they occur more frequently. Assigning unequal lengths requires a rule for deciding which objects receive the shorter ones. A probability distribution supplies that rule.

## Possibilities Have Different Probabilities

The counting bound treats every admissible object symmetrically. To assign shorter descriptions to some objects, we need a rule that determines which objects receive them and which objects pay with longer descriptions. A probability distribution supplies that rule.

Let $\mathcal{X}$ be a finite set of possible objects. For each $x\in\mathcal{X}$, a source specifies a probability

$$
P(x)=\Pr(X=x)
$$

where $P(x)>0$ and

$$
\sum_{x\in\mathcal{X}}P(x)=1
$$

A probability must be translated into a quantity measured in bits. If two independent outcomes occur with probabilities $P(x)$ and $P(y)$, their joint probability is the product $P(x)P(y)$, while their bit costs should add. The logarithm performs this conversion. The [information content](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) of an outcome $x$ is

$$
I_P(x) = -\log_2P(x)
$$

An event with probability $2^{-b}$ has information content $b$ bits. More probable outcomes receive smaller values because fewer bits should be allocated to events that occur more often.

Before the source produces an outcome, its information content is not known. Its expected value is

$$
\begin{aligned}
H(P) &= \sum_{x\in\mathcal{X}}P(x)I_P(x) \\
&= -\sum_{x\in\mathcal{X}}P(x)\log_2P(x)
\end{aligned}
$$

This is the [Shannon entropy](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) of the source. When $P$ is uniform, every object has information content

$$
\log_2|\mathcal{X}|
$$

so Shannon entropy equals the counting bound of the family.

The values $I_P(x)$ are ideal bit costs derived from the source distribution. To obtain a representation, we need a binary code

$$
C : \mathcal{X} \longrightarrow \{0,1\}^*
$$

with codeword lengths

$$
\ell_C(x) = |C(x)|
$$

The expected number of bits used by the code is

$$
L_P(C) = \sum_{x\in\mathcal{X}} P(x)\ell_C(x)
$$

The entropy $H(P)$ averages the real-valued lengths $-\log_2P(x)$ determined by the source. The expected code length $L_P(C)$ averages the integer lengths chosen by the encoder. To compare them, we must first determine which collections of integer lengths can belong to a decodable code.

Now, a probability distribution suggests ideal code lengths, but not every assignment of integer lengths can belong to a decodable code. The Kraft–McMillan inequality characterizes the available coding budget. Once that constraint has been established, the connection can also be run in reverse: the lengths of any uniquely decodable code can be turned into probability weights.

Assigning a different binary string to each object is sufficient when each codeword is presented in isolation. It is not sufficient when codewords are concatenated. A concatenated bitstream could admit two decompositions into codewords and therefore two possible source sequences. A code is [uniquely decodable](https://en.wikipedia.org/wiki/Variable-length_code) when every concatenation of its codewords has only one decomposition.

A [prefix code](https://en.wikipedia.org/wiki/Prefix_code) guarantees this property by requiring that no codeword be a prefix of another. Its codewords can be placed at leaves of a binary tree. Each edge contributes one bit, and the depth of a leaf equals the length of its codeword.

Let $m$ be at least as large as the longest codeword. A codeword of length $\ell_C(x)$ has

$$
2^{m-\ell_C(x)}
$$

descendants at depth $m$. Since no codeword is a prefix of another, the sets of descendants belonging to distinct codewords are disjoint. The complete binary tree contains only $2^m$ nodes at depth $m$, so

$$
\sum_{x\in\mathcal{X}} 2^{m-\ell_C(x)} \leq 2^m
$$

Dividing by $2^m$ gives

$$
\sum_{x\in\mathcal{X}} 2^{-\ell_C(x)} \leq 1
$$

This is the [Kraft–McMillan inequality](https://en.wikipedia.org/wiki/Kraft%E2%80%93McMillan_inequality). The tree argument proves it for prefix codes. The same inequality is necessary for every uniquely decodable code, even when its codewords do not form leaves of a prefix tree. Conversely, any collection of non-negative integer lengths satisfying the inequality can be realized by a prefix code.

The diagram in [_Reinventing Entropy_](https://youtu.be/l6DKRf-fAAM?t=518) visualizes the prefix-code case of this inequality. A codeword of length $\ell$ excludes all of its descendants and therefore occupies a fraction $2^{-\ell}$ of the available binary coding space. The video uses this geometry to explain why assigning one symbol a shorter codeword leaves less space for the others; Kraft–McMillan makes the same constraint precise and extends it to every uniquely decodable code.

The source probabilities satisfy a related identity. From the definition of information content,

$$
2^{-I_P(x)} = P(x)
$$

and therefore

$$
\sum_{x\in\mathcal{X}} 2^{-I_P(x)} = \sum_{x\in\mathcal{X}} P(x) = 1
$$

The ideal lengths $I_P(x)$ satisfy the same constraint as codeword lengths, with equality. They may nevertheless be fractional, so they do not necessarily specify a binary code.

The connection also runs in the opposite direction. Given a uniquely decodable code $C$, define its Kraft sum

$$
S_C = \sum_{x\in\mathcal{X}} 2^{-\ell_C(x)}
$$

The Kraft–McMillan inequality gives $S_C\leq1$. The quantities $2^{-\ell_C(x)}$ may therefore sum to less than one, but normalization turns them into a probability distribution:

$$
Q_C(x) = \frac{2^{-\ell_C(x)}}{S_C}
$$

Solving for the codeword length gives

$$
\ell_C(x) = -\log_2Q_C(x)-\log_2S_C
$$

The code lengths are therefore information contents under the induced distribution $Q_C$, shifted by the same non-negative amount $-\log_2S_C$. When the Kraft inequality is tight, so that $S_C=1$, the correspondence is exact:

$$
\ell_C(x) = -\log_2Q_C(x)
$$

A fixed-length code is the simplest instance. If every object receives the same length, then every value $2^{-\ell_C(x)}$ is equal, and normalization produces the uniform distribution over the objects. A non-uniform code induces a non-uniform distribution in which shorter codewords correspond to more probable objects.

This does not mean that a probability distribution was required to define the code. The counting argument in the previous section produced a fixed-length code without assuming a source. It means that once a uniquely decodable code has been chosen, its lengths can always be given a probabilistic interpretation. Codes constrain probability assignments, and probability assignments suggest code lengths.

The induced distribution also proves that entropy lower-bounds expected code length. Averaging

$$
\ell_C(x) = -\log_2Q_C(x)-\log_2S_C
$$

under the source distribution gives

$$
L_P(C) = -\sum_{x\in\mathcal{X}} P(x)\log_2Q_C(x) -\log_2S_C
$$

Subtracting the source entropy yields

$$
\begin{aligned}
L_P(C)-H(P) &= \sum_{x\in\mathcal{X}} P(x) \log_2 \frac{P(x)}{Q_C(x)} -\log_2S_C \\
&= D_{\mathrm{KL}}\left(P\mathbin\Vert Q_C\right) -\log_2S_C
\end{aligned}
$$

Both terms are non-negative. The first is the Kullback–Leibler divergence from the source distribution to the distribution induced by the code. The second is non-negative because $S_C\leq1$. Therefore every uniquely decodable code satisfies

$$
L_P(C) \geq H(P)
$$

The excess length has two sources. The divergence measures how poorly the code lengths match the source probabilities. The term $-\log_2S_C$ measures unused coding capacity when the Kraft inequality is not tight.

It remains to determine whether the entropy bound can be approached. Round each ideal length upward:

$$
\ell(x) = \left\lceil -\log_2P(x) \right\rceil
$$

Since

$$
\ell(x) \geq -\log_2P(x)
$$

we have

$$
2^{-\ell(x)} \leq P(x)
$$

and therefore

$$
\sum_{x\in\mathcal{X}} 2^{-\ell(x)} \leq \sum_{x\in\mathcal{X}} P(x) = 1
$$

The rounded lengths satisfy the Kraft–McMillan inequality, so a prefix code with those lengths exists. They also satisfy

$$
-\log_2P(x) \leq \ell(x) < -\log_2P(x)+1
$$

Averaging gives

$$
H(P) \leq L_P(C) < H(P)+1
$$

Together, the lower bound and this construction give the one-symbol form of [Shannon’s Source Coding Theorem](https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem). No uniquely decodable binary code has expected length below the entropy of a known source, while a prefix code can always remain within one bit of it.

For $r$ independent outcomes drawn from $P$, apply the same construction to the product distribution $P^r$. Its entropy is

$$
H(P^r)=rH(P)
$$

and a prefix code exists whose expected block length $L_r$ satisfies

$$
rH(P)\leq L_r<rH(P)+1.
$$

Dividing by $r$ gives

$$
H(P)\leq \frac{L_r}{r}<H(P)+\frac{1}{r}.
$$

The rounding overhead per source symbol therefore approaches zero as the block length grows. More generally, for dependent outcomes the same argument applies to their joint block distribution, with $H(P^r)$ replaced by the block entropy.

Up to this point, each value $x\in\mathcal{X}$ has been treated as a complete object. The equivalence between code lengths and probability assignments is already present, but no next symbol has appeared. Prediction enters only after an object is represented as an ordered sequence and its probability is factored into conditional probabilities for the successive symbols.

## When Compression Becomes Prediction

Now suppose that the object is an ordered sequence

$$
x_{1:n}=x_1,\ldots,x_n\in\Sigma^n
$$

over an alphabet $\Sigma$. Write

$$
x_{<i}=x_1,\ldots,x_{i-1}
$$

for the prefix before position $i$. The probability of the complete sequence satisfies the chain rule

$$
P(x_{1:n}) = \prod_{i=1}^n P(x_i\mid x_{<i})
$$

No independence assumption is involved. Each conditional distribution may depend on the complete prefix.

Applying $-\log_2$ turns the product into a sum:

$$
-\log_2P(x_{1:n}) = \sum_{i=1}^n -\log_2P(x_i\mid x_{<i})
$$

The information content of the complete sequence is therefore the sum of the conditional information contents of its symbols. Averaging over all possible sequences gives

$$
H(X_{1:n}) = \sum_{i=1}^n H(X_i\mid X_{<i})
$$

where

$$
H(X_i\mid X_{<i}) = \mathbb{E}\left[-\log_2P(X_i\mid X_{<i})\right]
$$

measures the information that remains at position $i$ after the prefix is known.

A compressor rarely knows the true conditional distributions. It instead uses a model $Q$ that returns, for every prefix, a distribution

$$
Q(\cdot\mid x_{<i})
$$

over $\Sigma$, with

$$
\sum_{a\in\Sigma}Q(a\mid x_{<i})=1
$$

Returning only the most likely symbol is not sufficient. The encoder already knows which symbol occurs and needs a code length for that symbol, whether or not it was the model’s first choice. Two models may select the same most likely continuation while assigning different probabilities to the observed symbol.

The model defines a probability for the complete sequence:

$$
Q(x_{1:n}) = \prod_{i=1}^n Q(x_i\mid x_{<i})
$$

Its cumulative logarithmic loss is

$$
\begin{aligned}
\mathcal{L}_Q(x_{1:n}) &= \sum_{i=1}^n -\log_2Q(x_i\mid x_{<i}) \\
&= -\log_2Q(x_{1:n})
\end{aligned}
$$

A high probability assigned to the observed symbol produces a small loss. A low probability produces a large loss. A model used for lossless coding must assign positive probability to every symbol that may occur, since probability zero would give an infinite code length.

During generation, a model chooses or samples a symbol from this distribution. During compression, the actual symbol is already known. The distribution is used to determine how much of the code space that symbol receives. Prediction in this equivalence means assigning probabilities, not guessing one continuation and replacing the data with that guess.

The model still does not produce a bitstream. An entropy coder must convert its probability assignments into a decodable representation. [Arithmetic coding](https://en.wikipedia.org/wiki/Arithmetic_coding) begins with the interval $[0,1)$. At position $i$, it partitions the current interval into adjacent subintervals whose widths are proportional to

$$
Q(\cdot\mid x_{<i})
$$

and retains the subinterval assigned to the observed symbol $x_i$.

If the current interval has width $w_{i-1}$, the selected interval has width

$$
w_i = w_{i-1}Q(x_i\mid x_{<i})
$$

Starting from $w_0=1$, the final width is

$$
w_n = \prod_{i=1}^n Q(x_i\mid x_{<i}) = Q(x_{1:n})
$$

The emitted binary prefix identifies a dyadic interval. To decode the sequence unambiguously, that dyadic interval must be contained in the final arithmetic-coding interval, together with an agreed termination convention. The number of required bits is therefore

$$
-\log_2Q(x_{1:n})
$$

plus a bounded coding overhead in the ideal arithmetic-coding model.

Decoding repeats the same subdivisions. After recovering $x_{<i}$, the decoder evaluates the same distribution $Q(\cdot\mid x_{<i})$, partitions its interval in the same order, and determines which subinterval contains the encoded value. Encoder and decoder must begin from the same state, perform the same updates, use the same symbol ordering, and agree on where the sequence ends.

This proves one direction of the equivalence. A sequential probabilistic model can be converted into a lossless compressor whose payload length follows the model’s log-loss. The [ngrok article](https://ngrok.com/blog/compression-is-prediction) and the two 3Blue1Brown videos ([_Reinventing Entropy_](https://youtu.be/l6DKRf-fAAM) and [_But what is cross-entropy?_](https://youtu.be/GlYgs6v2YfU)) explain this direction through entropy, cross-entropy, and conditional prediction.

The converse also holds. A uniquely decodable compressor assigns lengths to complete strings, and those lengths induce probability weights. After normalization, the resulting distribution can be factored into next-symbol conditionals. [_Language Modeling Is Compression_](https://arxiv.org/html/2309.10668v2) also constructs predictors directly from changes in compressed length when candidate symbols are appended to a prefix.

The equivalence is therefore not limited to compressors explicitly implemented as a probability model followed by arithmetic coding. At a mathematical level, codes and probability assignments can be translated into one another. This generality also limits what the equivalence tells us. It applies after the possible objects, their serialization, and the information available to the decoder have been fixed. It does not determine any of them.

The log-loss measures the data encoded under $Q$. It does not yet account for how $Q$ was chosen or how the decoder obtains it.

## The Source Is Unknown

The log-loss identity derived above measures the data under an already available model. In practice, that model must be estimated from the observed sequence, transmitted, or learned through a procedure the decoder can reproduce.

This section reaches the resulting description length in two ways. Maximum likelihood gives the best in-sample log-loss within a fixed model family. Counting type classes gives nearly the same length without assuming that the sequence was sampled from that model. The lower-order gap is bounded by the information needed to identify the fitted type.

Let

$$
S=s_1,\ldots,s_n\in\Sigma^n,\qquad n\geq 1
$$

be the observed sequence. Consider first the family of zero-order models, which use the same distribution at every position and ignore the preceding symbols. Let $q(a)$ be the probability assigned to a symbol $a\in\Sigma$. The model assigns the complete sequence the probability

$$
q(S) = \prod_{i=1}^n q(s_i)
$$

If $n_a$ denotes the number of occurrences of $a$ in $S$, equal factors can be collected:

$$
q(S) = \prod_{a\in\Sigma} q(a)^{n_a}
$$

The corresponding logarithmic loss is

$$
-\log_2q(S) = \sum_{a\in\Sigma} n_a\log_2\frac{1}{q(a)}
$$

Once the sequence has been observed, the counts determine which distribution in this model family assigns it the smallest loss. Define the empirical distribution

$$
\widehat{P}_S(a) = \frac{n_a}{n}
$$

For any distribution $q$ that assigns positive probability to every symbol occurring in $S$,

$$
\begin{aligned}
-\log_2q(S) &= \sum_{a\in\Sigma} n_a\log_2\frac{1}{q(a)} \\
&= n\sum_{a\in\Sigma} \widehat{P}_S(a) \log_2\frac{1}{q(a)}
\end{aligned}
$$

Under the empirical distribution, the probability assigned to the complete sequence is

$$
\widehat{P}_S(S) := \prod_{i=1}^n \widehat{P}_S(s_i)
$$

Its logarithmic loss is

$$
-\log_2\widehat{P}_S(S) = n\sum_{a\in\Sigma} \widehat{P}_S(a) \log_2 \frac{1}{\widehat{P}_S(a)}
$$

Subtracting the two quantities gives

$$
\begin{aligned}
-\log_2q(S) + \log_2\widehat{P}_S(S) &= n\sum_{a\in\Sigma} \widehat{P}_S(a) \log_2 \frac{\widehat{P}_S(a)}{q(a)} \\
&= nD_{\mathrm{KL}}\left(\widehat{P}_S \mathbin\Vert q\right) \\
&\geq 0
\end{aligned}
$$

Terms with $\widehat{P}_S(a)=0$ contribute zero. Since the divergence is non-negative, no zero-order distribution assigns the observed sequence a smaller log-loss than $\widehat{P}_S$. Equality holds exactly when

$$
q=\widehat{P}_S
$$

The empirical frequencies are therefore the maximum-likelihood estimate within the family of zero-order models. Equivalently, they minimize the in-sample logarithmic loss over that family.

The resulting cost per symbol is the [zero-order empirical entropy](https://arxiv.org/abs/0708.2084):

$$
\mathcal{H}_0(S) = \sum_{a\in\Sigma} \frac{n_a}{n} \log_2\frac{n}{n_a}
$$

with the convention that terms for which $n_a=0$ contribute zero. Multiplying by $n$ gives

$$
n\mathcal{H}_0(S) = \sum_{a\in\Sigma} n_a\log_2\frac{n}{n_a} = -\log_2\widehat{P}_S(S)
$$

Unlike Shannon entropy, $\mathcal{H}_0(S)$ is not defined from a source distribution that exists independently of the data. It is a property of the individual sequence $S$, obtained by fitting a zero-order model to its observed symbol frequencies. It does not assert that the sequence was generated by independent draws from that distribution.

A closely related description length can be obtained without beginning from a probabilistic model. More precisely, the counting argument recovers the same leading term, with a logarithmic difference that we will bound explicitly. Consider a binary string $B$ of length $n$ containing exactly $m$ ones. If the decoder knows $n$ and $m$, then $B$ belongs to the family

$$
\mathcal{B}_{n,m} = \left\{ B\in\{0,1\}^n : B\text{ contains exactly }m\text{ ones} \right\}
$$

A member of this family is determined by choosing which $m$ positions contain a one, so

$$
|\mathcal{B}_{n,m}| = \binom{n}{m}
$$

The counting bound from the first section says that identifying an arbitrary member of this family requires

$$
\log_2\binom{n}{m}
$$

bits, up to integer rounding.

For $0<m<n$, set

$$
p=\frac{m}{n}
$$

The endpoint cases $m=0$ and $m=n$ contain only one binary string and have both counting bound and empirical entropy equal to zero. Under the zero-order model that assigns probability $p$ to a one, every member of $\mathcal{B}_{n,m}$ receives the same probability:

$$
\begin{aligned}
p^m(1-p)^{n-m} &= \left(\frac{m}{n}\right)^m \left(\frac{n-m}{n}\right)^{n-m} \\
&= 2^{-n\mathcal{H}_0(B)}
\end{aligned}
$$

The total probability assigned to the family is therefore

$$
\binom{n}{m} 2^{-n\mathcal{H}_0(B)}
$$

Since this probability cannot exceed one,

$$
\binom{n}{m} 2^{-n\mathcal{H}_0(B)} \leq 1
$$

and hence

$$
\log_2\binom{n}{m} \leq n\mathcal{H}_0(B)
$$

For the reverse bound, consider the number of ones produced by the fitted Bernoulli model. This count can take only the $n+1$ values from $0$ to $n$. When $p=m/n$, the count $m$ is a mode of the resulting binomial distribution. Its probability is therefore at least the average probability of the possible counts:

$$
\binom{n}{m} p^m(1-p)^{n-m} \geq \frac{1}{n+1}
$$

Substituting the expression in terms of empirical entropy gives

$$
\binom{n}{m} 2^{-n\mathcal{H}_0(B)} \geq \frac{1}{n+1}
$$

Taking logarithms yields

$$
n\mathcal{H}_0(B) - \log_2(n+1) \leq \log_2\binom{n}{m}
$$

Together,

$$
n\mathcal{H}_0(B) - \log_2(n+1) \leq \log_2\binom{n}{m} \leq n\mathcal{H}_0(B)
$$

The counting bound and the best zero-order log-loss differ by at most $\log_2(n+1)$ bits. The counting argument conditions on $m$ and distinguishes the binary strings that remain possible. The probabilistic argument distributes probability across binary strings with every possible number of ones, then evaluates $B$ under the model fitted from its own count.

If $n$ is known but $m$ is not, the value of $m$ must also be represented. There are $n+1$ possible values, so a fixed-width representation uses

$$
\left\lceil \log_2(n+1) \right\rceil
$$

bits. Once this cost is included, the counting and probabilistic descriptions agree within the same lower-order term.

The relation extends to a general alphabet. Let the composition of $S$ be the vector

$$
(n_a)_{a\in\Sigma}
$$

and consider its type class

$$
\mathcal{T}(n_a) = \left\{ T\in\Sigma^n : T\text{ contains exactly }n_a\text{ occurrences of each }a\in\Sigma \right\}
$$

where $\sum_{a\in\Sigma}n_a=n$. A sequence in this class is obtained by choosing which positions contain each symbol, so

$$
|\mathcal{T}(n_a)| = \frac{n!}{\prod_{a\in\Sigma}n_a!}
$$

Every sequence in this type class receives the same probability under the empirical distribution:

$$
\begin{aligned}
\widehat{P}_S(T) &= \prod_{\substack{a\in\Sigma \\ n_a>0}} \left(\frac{n_a}{n}\right)^{n_a} \\
&= 2^{-n\mathcal{H}_0(S)}
\end{aligned}
$$

The total probability assigned to the type class is

$$
|\mathcal{T}(n_a)| 2^{-n\mathcal{H}_0(S)}
$$

Since this probability cannot exceed one,

$$
|\mathcal{T}(n_a)| \leq 2^{n\mathcal{H}_0(S)}
$$

and therefore

$$
\log_2 \frac{n!}{\prod_{a\in\Sigma}n_a!} \leq n\mathcal{H}_0(S)
$$

For the reverse direction, there are at most

$$
(n+1)^{|\Sigma|}
$$

possible type vectors. Under the multinomial distribution $\widehat{P}_S$, the observed count vector $(n_a)_{a\in\Sigma}$ is a mode. Its probability must therefore be at least the reciprocal of the number of possible types:

$$
|\mathcal{T}(n_a)|2^{-n\mathcal{H}_0(S)} \geq \frac{1}{(n+1)^{|\Sigma|}}
$$

Rearranging gives

$$
|\mathcal{T}(n_a)| \geq \frac{2^{n\mathcal{H}_0(S)}}{(n+1)^{|\Sigma|}}
$$

Taking logarithms produces

$$
n\mathcal{H}_0(S) - |\Sigma|\log_2(n+1) \leq \log_2 \frac{n!}{\prod_{a\in\Sigma}n_a!} \leq n\mathcal{H}_0(S)
$$

This is the general form of the [method of types](https://ieeexplore.ieee.org/document/720546/). Equivalently, there exists a quantity $\Delta(S)$ such that

$$
\log_2 \frac{n!}{\prod_{a\in\Sigma}n_a!} = n\mathcal{H}_0(S) - \Delta(S)
$$

where

$$
0 \leq \Delta(S) \leq |\Sigma|\log_2(n+1)
$$

For a fixed alphabet, the previous bound says simply that

$$
\Delta(S)=O(\log n).
$$

Thus the type-class description and the maximum-likelihood zero-order log-loss have the same linear term, while their difference grows at most logarithmically.

The two constructions condition on the same empirical information in different ways. The counting argument first fixes the composition and then counts the sequences that remain possible. The probabilistic argument fits the maximum-likelihood zero-order distribution and evaluates the observed sequence under it. Their difference is bounded by the information needed to identify the type.

Both quantities depend only on the composition of $S$. Reordering the symbols leaves every $n_a$ unchanged and therefore leaves both the type class and $\mathcal{H}_0(S)$ unchanged. A zero-order model assigns the same probability to a symbol wherever it occurs, even when the preceding symbols make some continuations more likely than others.

Fix a context length $0\leq k<n$. To introduce context without assuming a known source, group positions according to the symbols that precede them. For $a\in\Sigma$ and $\omega\in\Sigma^k$, define

$$
n_{\omega a} = \left| \left\{ i\in\{k+1,\ldots,n\} : s_{i-k:i-1}=\omega,\ s_i=a \right\} \right|
$$

and let

$$
n_\omega = \sum_{a\in\Sigma} n_{\omega a}
$$

be the number of symbols observed after $\omega$.

Within this group, the empirical conditional distribution is

$$
\widehat{P}_S(a\mid\omega) = \frac{n_{\omega a}}{n_\omega}
$$

for every observed context with $n_\omega>0$. The same maximum-likelihood argument used in the zero-order case applies independently to each context. The smallest log-loss obtained by assigning one distribution to the symbols following $\omega$ is

$$
\sum_{\substack{a\in\Sigma \\ n_{\omega a}>0}} n_{\omega a} \log_2 \frac{n_\omega}{n_{\omega a}}
$$

Summing over the observed contexts gives

$$
n\mathcal{H}_k(S) = \sum_{\substack{\omega\in\Sigma^k \\ n_\omega>0}} \sum_{\substack{a\in\Sigma \\ n_{\omega a}>0}} n_{\omega a} \log_2 \frac{n_\omega}{n_{\omega a}}
$$

This defines the [$k$-th order empirical entropy](https://arxiv.org/abs/0708.2084) under the boundary convention that only positions with a complete length-$k$ context contribute to the sum.

Equivalently, let $S_\omega$ be the sequence formed by collecting, in their original order, all symbols that follow occurrences of $\omega$. Then

$$
\mathcal{H}_k(S) = \frac{1}{n} \sum_{\substack{\omega\in\Sigma^k \\ n_\omega>0}} |S_\omega| \mathcal{H}_0(S_\omega)
$$

Each $S_\omega$ is itself a sequence over $\Sigma$. The zero-order analysis applies separately to it. Its empirical distribution is the maximum-likelihood model for the symbols observed after $\omega$, while its multinomial type class counts the alternative sequences with the same conditional composition.

The first $k$ symbols have no complete length-$k$ context. They may be encoded separately as one block in

$$
\left\lceil \log_2|\Sigma|^k \right\rceil = \left\lceil k\log_2|\Sigma| \right\rceil
$$

bits, or handled through an agreed boundary convention. When $k=0$, there is only the empty context and the definition reduces to $\mathcal{H}_0(S)$.

Longer contexts divide the observed positions into smaller groups. Refining a group cannot increase the minimum fitted log-loss when both models are evaluated on the same set of positions, since the refined model can always reuse the distribution of the original group. More context can therefore reduce the empirical data term.

This reduction eventually exposes a limitation of the measure. If every observed context is followed by only one distinct symbol, then every $S_\omega$ is constant and

$$
\mathcal{H}_0(S_\omega)=0
$$

for all observed contexts. Consequently,

$$
\mathcal{H}_k(S)=0
$$

This always occurs at $k=n-1$, where only one position has a complete context, and it may occur much earlier when sufficiently long contexts determine their observed continuations.

The value $\mathcal{H}_k(S)=0$ does not give a zero-length lossless representation of $S$. It gives a zero data term after the fitted context distributions are available. A decoder can assign probability one to an observed continuation only if it already knows which continuation followed that context. As the empirical loss decreases, information may have moved from the encoded sequence into the fitted model. The complete description must account for how the decoder obtains that model.

## The Model Is Part of the Message

That missing information is the model itself.

A naïve dense order-$k$ table contains one row for each of the $\sigma^k$ contexts and one count for each possible continuation. Using one $\lceil\log_2(n+1)\rceil$-bit field per count requires

$$
\sigma^{k+1}\left\lceil\log_2(n+1)\right\rceil = O\left(\sigma^{k+1}\log(n+1)\right)
$$

bits. A sparse representation removes entries for contexts that never occur, but it must still identify the observed contexts and their continuations. Increasing $k$ can reduce the empirical data term by moving more information into this structure.

Let $M$ contain everything the decoder needs to reproduce the probabilities used by the encoder. A two-part description has length

$$
L(M,S) = L(M)+L(S\mid M)
$$

The term $L(M)$ describes the model. The term $L(S\mid M)$ encodes the sequence using that model. In a sufficiently expressive model family, minimizing only the second term rewards memorization. A more complex model improves the complete description only when the reduction in $L(S\mid M)$ exceeds the additional cost of describing $M$. This is the basic two-part form of the [minimum description length principle](https://doi.org/10.7551/mitpress/1114.003.0005).

The data term also separates uncertainty in the source from mismatch in the model. Suppose objects are generated according to a distribution $P$, while the compressor assigns probabilities according to $Q$. If $Q(x)>0$ whenever $P(x)>0$, the expected ideal data length is the cross-entropy

$$
H(P,Q) = -\sum_x P(x)\log_2 Q(x)
$$

Subtracting the source entropy gives

$$
\begin{aligned}
H(P,Q)-H(P) &= \sum_x P(x)\log_2\frac{P(x)}{Q(x)} \\
&= D_{\mathrm{KL}}(P\mathbin\Vert Q)
\end{aligned}
$$

and therefore

$$
H(P,Q) = H(P)+D_{\mathrm{KL}}(P\mathbin\Vert Q)
$$

For sequential distributions, relative entropy decomposes across positions:

$$
D_{\mathrm{KL}}\left(P_{1:n}\mathbin\Vert Q_{1:n}\right) = \sum_{i=1}^n \mathbb{E}_{X_{<i}\sim P} \left[ D_{\mathrm{KL}} \left( P(\cdot\mid X_{<i}) \mathbin\Vert Q(\cdot\mid X_{<i}) \right) \right]
$$

Each term is the expected number of additional bits paid at one position because the model’s conditional distribution differs from the source distribution.

This distinction matters when discussing whether a better predictor “reduces entropy.” For a fixed source $P$, improving $Q$ under expected logarithmic loss means reducing the cross-entropy $H(P,Q)$, equivalently reducing the mismatch term $D_{\mathrm{KL}}(P\mathbin\Vert Q)$. It does not change $H(P)$. The [ngrok article](https://ngrok.com/blog/compression-is-prediction) correctly associates better probability estimates with shorter encodings, but its final use of _entropy_ merges these two quantities.

Whether $L(M)$ belongs to each transmitted file depends on the accounting boundary. If the model is fixed by a file format, built into the decoder, or otherwise shared in advance, it does not belong to the conditional description length of an individual message. It remains part of the system that makes that description meaningful, but charging its complete size to every message would also be misleading.

In an offline two-part code, if the fitted model is not already available to the decoder, a description of its tables, parameters, or weights must accompany the encoded data. [_Language Modeling Is Compression_](https://arxiv.org/html/2309.10668v2#S3.SS2) calls the ratio obtained without parameter size the _raw compression rate_. Its _adjusted compression rate_ adds the parameter size to the compressed output. A larger model may obtain a lower log-loss while producing a worse adjusted rate when it is amortized over too little data.

A third option is prequential or online coding. Encoder and decoder begin from the same initial state. Let $Q_{i-1}$ be the model available after the prefix $s_{<i}$ has been processed. The ideal prequential length is

$$
L_{\mathrm{preq}}(S) = \sum_{i=1}^n -\log_2 Q_{i-1}(s_i\mid s_{<i})
$$

After decoding $s_i$, the decoder performs the same update as the encoder and reconstructs $Q_i$. The final parameters therefore need not be transmitted.

The model cost is paid through the online log-loss rather than through a separate description of the final parameters. Before sufficient data have been observed, the current model will typically predict less well than a model fitted to the complete sequence. More generally, the difference appears as prequential regret relative to that offline fit. The initialization, update rule, training procedure, numerical conventions, and any randomness affecting them must be shared or described. Any unshared information needed to reproduce the learning procedure must be added to the prequential length.

## The Shortest Bitstream May Be the Wrong Representation

Every code considered so far has been judged by one operation: reconstructing the complete object. The two-part description length says nothing about what can be done with the encoded data before that reconstruction is complete.

Consider a vector

$$
A=(a_0,\ldots,a_{n-1})
$$

whose values belong to $\{0,\ldots,u-1\}$. Let

$$
b=\lceil\log_2 u\rceil
$$

A bit-packed representation assigns exactly $b$ consecutive bits to each value, using $nb$ bits for the payload apart from alignment and metadata. The representation of $a_i$ begins at bit position $ib$. If the storage-word width is $w\geq b$, recovering $a_i$ requires reading at most two adjacent words, shifting their contents, and applying a mask. The addresses and shifts are computed directly from $i$, so access takes $O(1)$ time.

This representation does not exploit differences in frequency. Every value receives the same number of bits. If the values follow a non-uniform distribution, or if their probabilities depend on earlier values, an entropy coder may produce a shorter stream:

$$
L(A\mid M) \approx \sum_{i=0}^{n-1} -\log_2 Q(a_i\mid a_{<i})
$$

The shorter stream provides a different access contract. In an ordinary arithmetic-coded stream, the decoding state at position $i$ depends on the symbols that precede it. If the model also uses their context, its next distribution depends on the same prefix. Recovering $a_i$ requires decoding from the beginning of the stream or from an earlier checkpoint whose coding and model states have been stored.

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

Bitstream length alone answers only the bare reconstruction problem. A compressed representation must also encode enough structure for its required operations. The decoder contract now has three explicit parts: the objects it must distinguish, the information it already shares with the encoder, and the operations it must perform without full reconstruction. Prediction determines conditional code lengths inside this contract. It does not determine the contract itself.

## So, Is Compression Prediction?

The two 3Blue1Brown videos ([_Reinventing Entropy_](https://youtu.be/l6DKRf-fAAM) and [_But what is cross-entropy?_](https://youtu.be/GlYgs6v2YfU)) derive the connection between coding, entropy, and cross-entropy. The [ngrok article](https://ngrok.com/blog/compression-is-prediction) shows how a model supplies conditional probabilities and arithmetic coding turns them into a bitstream. [Salvatore Sanfilippo](https://youtu.be/UgRiVUce9sY) asks whether this makes prediction and compression the same concept. [_Language Modeling Is Compression_](https://arxiv.org/html/2309.10668v2) studies both directions of the equivalence and explicitly accounts for model parameters through raw, adjusted, and prequential compression rates.

Kolmogorov complexity makes this visible for an individual string, while the counting bound makes it visible for a finite family: a description-length problem can be posed before a next-symbol predictor exists.

What I felt these explanations left implicit was the compression problem that must be fixed before the equivalence becomes meaningful. The encoder and decoder need an agreed family of objects, a serialization, a boundary between transmitted and shared information, and a decoding contract. None of these choices is determined by next-symbol prediction.

Once those choices have been made, the equivalence is broad. A sequential probability model assigns ideal payload lengths through logarithmic loss. Conversely, every uniquely decodable code over a fixed object family induces the distribution

$$
Q_C(x)=\frac{2^{-\ell_C(x)}}{S_C},
$$

and its lengths satisfy

$$
\ell_C(x)=-\log_2Q_C(x)-\log_2S_C.
$$

After the objects have been serialized, the induced distribution can in turn be factored into next-symbol conditionals.

That generality is also the limit of the slogan _Compression is Prediction_. Recasting a representation probabilistically does not explain why its objects were chosen, whether its model must be transmitted, or which operations it supports. A model with lower log-loss can produce a larger complete file after its parameters are included. An entropy-coded stream can use fewer bits than a bit-packed vector while failing to provide constant-time access. An empirical entropy of zero can still leave the decoder without the model needed to reconstruct the sequence.

Compression is therefore prediction _after_ the coding problem has been fixed, and only at the level measured by the induced code lengths. For a shared sequential model under logarithmic loss, cumulative prediction error gives the ideal payload length up to coding overhead. It does not define what must be represented, what the decoder already knows, or what the representation must allow the decoder to do.
