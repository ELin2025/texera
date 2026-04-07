/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.texera.web.resource

import com.fasterxml.jackson.core.`type`.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import kong.unirest.Unirest

import javax.ws.rs._
import javax.ws.rs.core.{MediaType, Response}
import java.util.concurrent.ConcurrentHashMap

/**
 * REST resource that proxies the Hugging Face Hub API to list
 * models for the HuggingFace operator.
 *
 * Browse mode:  GET /api/huggingface/models?task=text-generation
 *   Fetches ALL models for the task from HF Hub (paginated internally),
 *   caches the full list server-side, and returns it.
 *
 * Search mode:  GET /api/huggingface/models?task=text-generation&search=bert
 *   Forwards the search query to HF Hub API (searches all models).
 */
@Path("/huggingface")
@Produces(Array(MediaType.APPLICATION_JSON))
class HuggingFaceModelResource {

  import HuggingFaceModelResource._

  @GET
  @Path("/models")
  def listModels(
      @QueryParam("task") @DefaultValue("text-generation") task: String,
      @QueryParam("search") search: String
  ): Response = {
    try {
      val hfToken = Option(System.getenv("HF_TOKEN")).getOrElse("")

      // ── Search mode: forward query to HF Hub, return results directly ──
      if (search != null && search.trim.nonEmpty) {
        return fetchSearchResults(task, search.trim, hfToken)
      }

      // ── Browse mode: return ALL models for this task (cached) ──
      val cached = modelCache.get(task)
      if (cached != null) {
        return Response.ok(cached).build()
      }

      // Not cached — fetch all pages from HF Hub API
      val allModels = fetchAllModelsForTask(task, hfToken)
      val json = objectMapper.writeValueAsString(allModels)
      modelCache.put(task, json)

      Response.ok(json).build()

    } catch {
      case e: Exception =>
        Response
          .status(Response.Status.INTERNAL_SERVER_ERROR)
          .entity(s"""{"error":"Failed to fetch models: ${e.getMessage}"}""")
          .build()
    }
  }

  /** Search HF Hub for models matching a query within a task. */
  private def fetchSearchResults(task: String, query: String, hfToken: String): Response = {
    var request = Unirest
      .get("https://huggingface.co/api/models")
      .queryString("pipeline_tag", task)
      .queryString("sort", "downloads")
      .queryString("direction", "-1")
      .queryString("limit", "100")
      .queryString("filter", task)
      .queryString("search", query)

    if (hfToken.nonEmpty) {
      request = request.header("Authorization", s"Bearer $hfToken")
    }

    val hfResponse = request.asString()

    if (hfResponse.getStatus != 200) {
      return Response
        .status(hfResponse.getStatus)
        .entity(s"""{"error":"Hugging Face API error: ${hfResponse.getStatusText}"}""")
        .build()
    }

    val rawModels = objectMapper.readValue(hfResponse.getBody, listOfMapsType)
    val out = buildSimplifiedList(rawModels)
    Response.ok(objectMapper.writeValueAsString(out)).build()
  }

  /**
   * Fetch ALL models for a given task by paginating through the HF Hub API.
   * HF Hub uses a Link header with rel="next" for pagination.
   * We fetch pages of 1000 models at a time, up to MAX_PAGES pages.
   */
  private def fetchAllModelsForTask(
      task: String,
      hfToken: String
  ): java.util.List[java.util.Map[String, Object]] = {
    val allResults = new java.util.ArrayList[java.util.Map[String, Object]]()
    var nextUrl: String = null
    var pageCount = 0

    // First request
    var request = Unirest
      .get("https://huggingface.co/api/models")
      .queryString("pipeline_tag", task)
      .queryString("sort", "downloads")
      .queryString("direction", "-1")
      .queryString("limit", PAGE_SIZE.toString)
      .queryString("filter", task)
      .connectTimeout(10000)
      .socketTimeout(30000)

    if (hfToken.nonEmpty) {
      request = request.header("Authorization", s"Bearer $hfToken")
    }

    var hfResponse = request.asString()

    if (hfResponse.getStatus != 200) {
      throw new RuntimeException(
        s"HF API returned ${hfResponse.getStatus}: ${hfResponse.getStatusText}"
      )
    }

    var rawModels = objectMapper.readValue(hfResponse.getBody, listOfMapsType)
    allResults.addAll(buildSimplifiedList(rawModels))
    pageCount += 1

    // Extract next page URL from Link header
    nextUrl = extractNextLink(hfResponse.getHeaders.getFirst("Link"))

    // Fetch remaining pages until exhausted
    while (nextUrl != null) {
      var nextRequest = Unirest
        .get(nextUrl)
        .connectTimeout(10000)
        .socketTimeout(30000)
      if (hfToken.nonEmpty) {
        nextRequest = nextRequest.header("Authorization", s"Bearer $hfToken")
      }

      hfResponse = nextRequest.asString()

      if (hfResponse.getStatus != 200) {
        // Stop paginating on error, return what we have so far
        return allResults
      }

      rawModels = objectMapper.readValue(hfResponse.getBody, listOfMapsType)
      allResults.addAll(buildSimplifiedList(rawModels))
      pageCount += 1

      nextUrl = extractNextLink(hfResponse.getHeaders.getFirst("Link"))
    }

    allResults
  }

  /**
   * Parse the Link header to extract the URL with rel="next".
   * Format: <https://huggingface.co/api/models?...>; rel="next"
   */
  private def extractNextLink(linkHeader: String): String = {
    if (linkHeader == null || linkHeader.isEmpty) return null

    val parts = linkHeader.split(",")
    for (part <- parts) {
      val trimmed = part.trim
      if (trimmed.contains("rel=\"next\"")) {
        val start = trimmed.indexOf('<')
        val end = trimmed.indexOf('>')
        if (start >= 0 && end > start) {
          return trimmed.substring(start + 1, end)
        }
      }
    }
    null
  }

  /** Convert raw HF model maps into simplified maps for the frontend. */
  private def buildSimplifiedList(
      rawModels: java.util.List[java.util.Map[String, Object]]
  ): java.util.List[java.util.Map[String, Object]] = {
    val out = new java.util.ArrayList[java.util.Map[String, Object]]()
    val iter = rawModels.iterator()
    while (iter.hasNext) {
      val model = iter.next()
      val id = if (model.get("id") != null) model.get("id").toString else ""
      val downloads: java.lang.Long = model.get("downloads") match {
        case n: java.lang.Number => n.longValue()
        case _                   => 0L
      }
      val likes: java.lang.Long = model.get("likes") match {
        case n: java.lang.Number => n.longValue()
        case _                   => 0L
      }
      val pipelineTag =
        if (model.get("pipeline_tag") != null) model.get("pipeline_tag").toString else ""

      val entry = new java.util.LinkedHashMap[String, Object]()
      entry.put("id", id)
      entry.put("label", id)
      entry.put("pipeline_tag", pipelineTag)
      entry.put("downloads", downloads)
      entry.put("likes", likes)
      out.add(entry)
    }
    out
  }
}

object HuggingFaceModelResource {
  private val objectMapper: ObjectMapper = new ObjectMapper()

  private val listOfMapsType =
    new TypeReference[java.util.List[java.util.Map[String, Object]]]() {}

  /** Server-side cache: task → JSON string of all models. Thread-safe. */
  private val modelCache = new ConcurrentHashMap[String, String]()

  /** Number of models to fetch per HF API page. */
  private val PAGE_SIZE = 1000

}
