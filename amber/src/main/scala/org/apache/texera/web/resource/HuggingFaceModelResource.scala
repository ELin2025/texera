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

/**
 * REST resource that proxies the Hugging Face Hub API to list
 * text-generation models for the HuggingFaceTextGen operator.
 *
 * This keeps the HF_TOKEN secret on the backend —
 * the Angular frontend never touches the Hub API directly.
 *
 * Endpoint: GET /api/huggingface/models?task=text-generation&limit=50
 */
@Path("/huggingface")
@Produces(Array(MediaType.APPLICATION_JSON))
class HuggingFaceModelResource {

  // Plain ObjectMapper without DefaultScalaModule so that JSON arrays/objects
  // deserialize into java.util collections instead of Scala collections.
  private val objectMapper: ObjectMapper = new ObjectMapper()

  // TypeReference for Jackson to deserialize a JSON array of maps correctly
  private val listOfMapsType =
    new TypeReference[java.util.List[java.util.Map[String, Object]]]() {}

  @GET
  @Path("/models")
  def listModels(
      @QueryParam("task") @DefaultValue("text-generation") task: String,
      @QueryParam("limit") @DefaultValue("100") limit: Int,
      @QueryParam("search") search: String
  ): Response = {
    try {
      val hfToken = Option(System.getenv("HF_TOKEN")).getOrElse("")

      // Build the request to the Hugging Face Hub API
      var request = Unirest
        .get("https://huggingface.co/api/models")
        .queryString("pipeline_tag", task)
        .queryString("sort", "downloads")
        .queryString("direction", "-1")
        .queryString("limit", limit.toString)
        .queryString("filter", task)

      // Add search query if provided (for typeahead search)
      if (search != null && search.trim.nonEmpty) {
        request = request.queryString("search", search.trim)
      }

      // Add authorization header if token is available
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

      // Parse the HF response — it's a JSON array of model objects.
      // Using an explicit TypeReference ensures Jackson produces java.util types.
      val rawModels: java.util.List[java.util.Map[String, Object]] =
        objectMapper.readValue(hfResponse.getBody, listOfMapsType)

      // Build a simplified JSON array for the frontend
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

      Response.ok(objectMapper.writeValueAsString(out)).build()

    } catch {
      case e: Exception =>
        Response
          .status(Response.Status.INTERNAL_SERVER_ERROR)
          .entity(s"""{"error":"Failed to fetch models: ${e.getMessage}"}""")
          .build()
    }
  }
}
