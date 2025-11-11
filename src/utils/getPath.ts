import { BLOG_PATH, UNIVERSITY_PATH } from "@/content.config";
import { slugifyStr } from "./slugify";

/**
 * Get full path of a blog post or university project
 * @param id - id of the post/project (aka slug)
 * @param filePath - the post/project full file location
 * @param collection - collection name: "blog" or "university"
 * @param includeBase - whether to include the base path ("/posts" or "/university") in return value
 * @returns post/project path
 */
export function getPath(
  id: string,
  filePath: string | undefined,
  collection: "blog" | "university" = "blog",
  includeBase = true
) {
  const basePath = collection === "blog" ? "/posts" : "/university";
  const collectionPath = collection === "blog" ? BLOG_PATH : UNIVERSITY_PATH;

  const pathSegments = filePath
    ?.replace(collectionPath, "")
    .split("/")
    .filter(path => path !== "") // remove empty string in the segments ["", "other-path"] <- empty string will be removed
    .filter(path => !path.startsWith("_")) // exclude directories start with underscore "_"
    .slice(0, -1) // remove the last segment_ file name_ since it's unnecessary
    .map(segment => slugifyStr(segment)); // slugify each segment path

  const returnBasePath = includeBase ? basePath : "";

  // Making sure `id` does not contain the directory
  const postId = id.split("/");
  const slug = postId.length > 0 ? postId.slice(-1) : postId;

  // If not inside the sub-dir, simply return the file path
  if (!pathSegments || pathSegments.length < 1) {
    return [returnBasePath, slug].join("/");
  }

  return [returnBasePath, ...pathSegments, slug].join("/");
}
