import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";

interface Tag {
  tag: string;
  tagName: string;
}

const getUniqueTags = (
  blogPosts: CollectionEntry<"blog">[],
  universityPosts?: CollectionEntry<"university">[]
) => {
  // Combine all posts from both collections
  const allPosts = [...blogPosts, ...(universityPosts || [])];

  const tags: Tag[] = allPosts
    .flatMap(post => post.data.tags)
    .map(tag => ({ tag: slugifyStr(tag), tagName: tag }))
    .filter(
      (value, index, self) =>
        self.findIndex(tag => tag.tag === value.tag) === index
    )
    .sort((tagA, tagB) => tagA.tag.localeCompare(tagB.tag));
  return tags;
};

export default getUniqueTags;
