import type { CollectionEntry } from "astro:content";
import { slugifyAll } from "./slugify";

const sortByDate = (
  posts: (CollectionEntry<"blog"> | CollectionEntry<"university">)[]
) => {
  return posts.sort(
    (a, b) =>
      Math.floor(
        new Date(b.data.modDatetime ?? b.data.pubDatetime).getTime() / 1000
      ) -
      Math.floor(
        new Date(a.data.modDatetime ?? a.data.pubDatetime).getTime() / 1000
      )
  );
};

const getPostsByTag = (
  posts: (CollectionEntry<"blog"> | CollectionEntry<"university">)[],
  tag: string
) =>
  sortByDate(
    posts.filter(post => slugifyAll(post.data.tags).includes(tag))
  );

export default getPostsByTag;
