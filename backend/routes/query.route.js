import express from "express";
import path from "path";
import pool from "../db.js";
import {
  backendLink,
  modelLink,
} from "../../config.js";
import axios from "axios";

const router = express.Router();

/**
 * query.route.js
 * - API endpoints that orchestrate Visual Question Answering (VQA), grounding,
 *   captioning and a comprehensive evaluation flow.
 * - This module:
 *   - Looks up chat/session metadata from the database (image URL, merged polys/classes)
 *   - Forwards requests to a model server (modelLink) for inference
 *   - Persists generated messages/results to the messages table
 *   - Returns normalized objects for the frontend to consume
 */

/* --------------------------------------------------------------------------- */
/* Visual Question Answering (VQA)                                             */
/* - Accepts: { chatId, query, queryType }                                     */
/* - Behavior: validates input, fetches chat metadata, forwards to the model   */
/*   server endpoint chosen by queryType, stores the returned text answer as   */
/*   a message record and updates the parent chat's updated_at timestamp.      */
/* --------------------------------------------------------------------------- */
router.post("/vqa", async (req, res) => {
  const { chatId, query, queryType } = req.body;
  if (!chatId || !query || !queryType)
    return res.status(400).json({ error: "chatId and query required" });

  // Load chat row to access image URL and any previously-computed merged polygons/classes
  const chatResult = await pool.query(
    "SELECT * FROM chats WHERE id=$1",
    [chatId]
  );
  if (chatResult.rows.length === 0)
    return res.status(404).json({ error: "Chat not found" });

  const imageUrl = chatResult.rows[0].image_url;
  const fileName = path.basename(imageUrl);

  let response;
  // Route the request to the appropriate model endpoint depending on the query type.
  // Each model endpoint expects a slightly different payload key for the instruction.
  if (queryType == "semantic") {
    response = await axios.post(`${modelLink}/semantic`, {
      image_id: fileName,
      image_url: imageUrl,
      sem_instr: query,
      img_type: chatResult.rows[0].img_type,
    });
  } else if (queryType == "binary") {
    response = await axios.post(`${modelLink}/binary`, {
      image_id: fileName,
      image_url: imageUrl,
      img_type: chatResult.rows[0].img_type,
      bin_instr: query,
    });
  } else {
    // numeric queries may require geometry context (merged_polys / merged_cls) for counting/area measurements
    response = await axios.post(`${modelLink}/numeric_chat`, {
      image_id: fileName,
      image_url: imageUrl,
      num_instr: query,
      merged_polys: chatResult.rows[0].merged_polys,
      merged_cls: chatResult.rows[0].merged_cls,
      merged_source: chatResult.rows[0].merged_source,
      img_type: chatResult.rows[0].img_type,
      spatial_resolution: 0.0
    });
  }

  const data = response.data;
  const textAnswer = data.answer;

  // Persist the produced message/result into the messages table
  const result = await pool.query(
    "INSERT INTO messages (chat_id, query, text_answer, generated_image) VALUES ($1, $2, $3, $4) RETURNING *",
    [chatId, query, textAnswer, ""]
  );

  // Touch the chat to update last activity timestamp
  await pool.query("UPDATE chats SET updated_at = NOW() WHERE id = $1", [
    chatId,
  ]);

  // Return the newly inserted message row to the caller
  res.json(result.rows[0]);
});

/* ---------------------------------------------------------------------------- */
/* Grounding                                                                    */
/* - Accepts: { chatId, query }                                                 */
/* - Behavior: asks the model server to generate a grounded visualization (an   */
/*   output image). On success it writes a message record with a link to the    */
/*   generated artifact served via backend /api/results.                        */
/* ---------------------------------------------------------------------------- */
router.post("/grounding", async (req, res) => {
  const { chatId, query } = req.body;
  if (!chatId || !query)
    return res.status(400).json({ error: "chatId required" });

  // Ensure chat exists and read necessary metadata
  const chatResult = await pool.query(
    "SELECT * FROM chats WHERE id=$1",
    [chatId]
  );
  if (chatResult.rows.length === 0)
    return res.status(404).json({ error: "Chat not found" });

  const imageUrl = chatResult.rows[0].image_url;
  const fileName = path.basename(imageUrl);
  const outputPath = `${Date.now()}.jpg`;

  // Build model payload including merged geometry/class context if present
  const payload = {
    image_id: fileName,
    image_url: imageUrl,
    output_path: outputPath,
    grounding_instr: query,
    merged_polys: chatResult.rows[0].merged_polys,
    img_type: chatResult.rows[0].img_type,
    merged_cls: chatResult.rows[0].merged_cls,
    merged_source: chatResult.rows[0].merged_source,
  };

  // Call the model server and persist the produced artifact reference on success
  try {
    const response = await axios.post(`${modelLink}/grounding`, payload);

    const result = await pool.query(
      "INSERT INTO messages (chat_id, query, text_answer, generated_image) VALUES ($1, $2, $3, $4) RETURNING *",
      [
        chatId,
        "Grounding selected objects",
        "",
        `${backendLink}/api/results/${outputPath}`,
      ]
    );

    return res.json(result.rows[0]);
  } catch (err) {
    // Log model server errors (include body if available) and return 500
    console.error("Inference API error:", err.response?.data || err);
    return res.status(500).json({ error: "Inference failed" });
  }
});

/* ---------------------------------------------------------------------------- */
/* Captioning                                                                   */
/* - Accepts: { chatId }                                                        */
/* - Behavior: requests an image caption from the model server, updates the     */
/*   chats table with the caption text and returns it to the frontend.          */
/* ---------------------------------------------------------------------------- */
router.post("/caption", async (req, res) => {
  const { chatId } = req.body;
  if (!chatId) return res.status(400).json({ error: "chatId is required" });

  const chatResult = await pool.query(
    "SELECT * FROM chats WHERE id=$1",
    [chatId]
  );
  if (chatResult.rows.length === 0)
    return res.status(404).json({ error: "Chat not found" });

  const imageUrl = chatResult.rows[0].image_url;
  const fileName = path.basename(imageUrl);

  // Forward a descriptive caption instruction to the model server
  const response = await axios.post(`${modelLink}/caption`, { 
    image_id: fileName, 
    img_type: chatResult.rows[0].img_type,
    image_url: imageUrl, 
    cap_instr:
      "Generate a detailed caption describing all visible elements in the satellite image, including object types, counts, relative locations, and overall scene context."
  });

  const textAnswer = response.data.caption;

  // Persist the caption on the chats record and update timestamp
  await pool.query(
    "UPDATE chats SET updated_at = NOW(), caption = $1 WHERE id = $2",
    [textAnswer, chatId]
  );

  res.json({ caption: textAnswer });
});


function getDirectUrl(url) {
  // Google Drive file URL -> direct download
  if (url.includes("drive.google.com")) {
    const fileId = url.match(/\/d\/(.*?)\//)?.[1];
    if (fileId) {
      return `https://drive.google.com/uc?export=download&id=${fileId}`;
    }
  }

  // Dropbox share link
  if (url.includes("dropbox.com")) {
    return url.replace("?dl=0", "?dl=1");
  }

  // OneDrive "view" links
  if (url.includes("onedrive.live.com")) {
    return url.replace("redir?", "download?");
  }

  // Otherwise assume it is already a direct link
  return url;
}

/* --------------------------------------------------------------------------- */
/* Evaluation endpoint                                                         */
/* - Accepts a comprehensive payload that contains an input image (URL + id)   */
/*   and a set of queries (caption, grounding, attribute queries).             */
/* - Workflow: downloads the input image, uploads it via /api/upload to the    */
/*   backend (so the image is served consistently), sends it to the model      */
/*   server (upload endpoint) to obtain merged geometry outputs, then runs     */
/*   captioning/grounding/attribute queries and collates all responses into a  */
/*   single JSON output returned to the caller.                                */
/* - This endpoint is primarily for batch/evaluation usage and is more         */
/*   heavyweight than individual VQA/grounding routes.                         */
/* --------------------------------------------------------------------------- */
router.post("/evaluate", async (req, res) => {
  try {
    const payload = req.body;
    // Destructure expected payload structure for clarity
    const {
      input_image,
      queries: {
        caption_query,
        grounding_query,
        attribute_query: { binary, numeric, semantic },
      },
    } = payload;

    const { image_url, image_id, metadata } = input_image;

    if(!image_url){
      return res.status(400).json({ error: "No image provided" });
    }
    // Download remote image (arraybuffer) so we can re-upload it to our backend storage
    const directImageUrl = getDirectUrl(image_url);
    const imgResponse = await axios.get(directImageUrl, {
      responseType: "arraybuffer",
      maxRedirects: 5,
    });

    if (imgResponse.status !== 200) {
      return res.status(400).json({ error: "Image could not be downloaded" });
    }

    const buffer = Buffer.from(imgResponse.data);

    // Construct a File-like object and attach it to FormData so the /api/upload route can accept it.
    // Note: Node.js doesn't have a native File constructor; this mirrors the original intent.
    const file = new File([buffer], "input.jpg", { type: "image/jpeg" });
    const formData = new FormData();
    formData.append("image", file);

    // Upload to our backend so the model server can access the image via backendLink
    const imageUploadResponse = await axios.post(`${backendLink}/api/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    const imageUrl = imageUploadResponse.data.imageUrl;
    const imageId = path.basename(imageUrl);

    // Inform the model server about the image and receive merged polygon/class outputs
    const uploadResponse = await axios.post(`${modelLink}/upload`, { image_id: imageId, image_url: imageUrl });
    const img_type = uploadResponse.data.img_type;
    const merged_polys = uploadResponse.data.merged_polys;
    const merged_cls = uploadResponse.data.merged_cls;
    const merged_source = uploadResponse.data.merged_source;

    // Captioning
    let captionResponse;
    if(caption_query.instruction){
      captionResponse = await axios.post(`${modelLink}/caption`, {
        image_id: imageId,
        image_url: imageUrl,
        cap_instr: caption_query.instruction,
        img_type: img_type
      });
    }
    const captionResult = captionResponse.data.caption || "";

    // Grounding (produces an output image path on the model side)
    let groundingResponse;
    if(grounding_query.instruction){
      const groundingOutputFile = `${Date.now()}.jpg`;
      groundingResponse = await axios.post(`${modelLink}/grounding`, {
        image_id: imageId,
        image_url: imageUrl,
        grounding_instr: grounding_query.instruction,
        output_path: groundingOutputFile,
        merged_source: merged_source,
        merged_polys: merged_polys,
        merged_cls: merged_cls,
        img_type: img_type
      });
    }
    const groundingResult = groundingResponse.data.ground || [];

    // Attribute queries: binary, numeric and semantic
    let binaryResponse;
    let numericResponse;
    let semanticResponse;

    if(binary.instruction){
      binaryResponse = await axios.post(`${modelLink}/binary`, {
        image_id: imageId,
        image_url: imageUrl,
        bin_instr: binary.instruction,
        img_type: img_type
      });
    }

    if(numeric.instruction){
      numericResponse = await axios.post(`${modelLink}/numeric_evaluate`, {
        image_id: imageId,
        image_url: imageUrl,
        num_instr: numeric.instruction,
        merged_polys: merged_polys,
        merged_cls: merged_cls,
        merged_source: merged_source,
        spatial_resolution: metadata.spatial_resolution_m,
        img_type: img_type
      });
    }

    if(semantic.instruction){
      semanticResponse = await axios.post(`${modelLink}/semantic`, {
        image_id: imageId,
        image_url: imageUrl,
        sem_instr: semantic.instruction,
        img_type: img_type
      });
    }

    // Collate all responses into a single structured output
    const output = {
      input_image: {
        image_id,
        image_url,
        metadata,
      },
      queries: {
        caption_query: {
          instruction: caption_query.instruction,
          response: captionResult || "",
        },
        grounding_query: {
          instruction: grounding_query.instruction,
          response: groundingResult,
        },
        attribute_query: {
          binary: {
            instruction: binary.instruction,
            response: binaryResponse.data.answer || "",
          },
          numeric: {
            instruction: numeric.instruction,
            response: numericResponse.data.answer || "",
          },
          semantic: {
            instruction: semantic.instruction,
            response: semanticResponse.data.answer || "",
          },
        },
      },
    };

    res.json(output);
  } catch (err) {
    // Catch-all error reporting for this multi-step pipeline
    console.error("Evaluation Error:", err);
    res.status(500).json({ error: "Evaluation failed" });
  }
});

export default router;
