#include "model.hpp"

#include <utility>

#include <cstdio>
#include <cassert>

#include "../labutils/error.hpp"
namespace lut = labutils;

// ModelData
ModelData::ModelData() noexcept = default;

ModelData::ModelData( ModelData&& aOther ) noexcept
	: modelName( std::exchange( aOther.modelName, {} ) )
	, modelSourcePath( std::exchange( aOther.modelSourcePath, {} ) )
	, materials( std::move( aOther.materials ) )
	, meshes( std::move( aOther.meshes ) )
	, vertexPositions( std::move( aOther.vertexPositions ) )
	, vertexNormals( std::move( aOther.vertexNormals ) )
	, vertexTextureCoords( std::move( aOther.vertexTextureCoords ) )
{}

ModelData& ModelData::operator=( ModelData&& aOther ) noexcept
{
	std::swap( modelName, aOther.modelName );
	std::swap( modelSourcePath, aOther.modelSourcePath );
	std::swap( materials, aOther.materials );
	std::swap( meshes, aOther.meshes );
	std::swap( vertexPositions, aOther.vertexPositions );
	std::swap( vertexNormals, aOther.vertexNormals );
	std::swap( vertexTextureCoords, aOther.vertexTextureCoords );
	return *this;
}


// load_obj_model()
ModelData load_obj_model( std::string_view const& aOBJPath )
{
	// "Decode" path
	std::string fileName, directory;

	if( auto const separator = aOBJPath.find_last_of( "/\\" ); std::string_view::npos != separator )
	{
		fileName = aOBJPath.substr( separator+1 );
		directory = aOBJPath.substr( 0, separator+1 );
	}
	else
	{
		fileName = aOBJPath;
		directory = "./";
	}

	std::string const normalizedPath = directory + fileName;

	// Load model
	std::printf( "Loading: '%s' ...", normalizedPath.c_str() );
	std::fflush( stdout );

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	if( !tinyobj::LoadObj( &attrib, &shapes, &materials, &err, normalizedPath.c_str(), directory.c_str(), true ) )
	{
		throw lut::Error( "Unable to load OBJ '%s':\n%s", normalizedPath.c_str(), err.c_str() );
	}

	// Apparently this can include some warnings:
	if( !err.empty() )
		std::printf( "\n%s\n... OK", err.c_str() );
	else
		std::printf( " OK\n" );

	// Transfer into our ModelData structures
	ModelData model;
	model.modelName        = aOBJPath;
	model.modelSourcePath  = normalizedPath;

	// ... copy over material data ...
	for( auto const& m : materials )
	{
		MaterialInfo info{};
		info.materialName      = m.name;
		info.color             = glm::vec3( m.diffuse[0], m.diffuse[1], m.diffuse[2] );

		if( !m.diffuse_texname.empty() )
			info.colorTexturePath  = directory + m.diffuse_texname;

		model.materials.emplace_back( info );
	}

	// ... copy over mesh data ...
	// Note: this converts the mesh into a triangle soup. OBJ meshes use
	// separate indices for vertex positions, texture coordinates and normals.
	// This is not compatible with the default draw modes of OpenGL or Vulkan,
	// where each vertex has a single index that refers to all attributes.
	//
	// tinyobjloader additionally complicates the situation by specifying a
	// per-face material indices, which is rather impractical.
	//
	// In short- The OBJ format isn't exactly a great format (in a modern
	// context), and tinyobjloader is not making the situation a lot better.
	std::size_t totalVertices = 0;
	for( auto const& s : shapes )
	{
		totalVertices += s.mesh.indices.size();
	}

	model.vertexPositions.reserve( totalVertices );
	model.vertexNormals.reserve( totalVertices );
	model.vertexTextureCoords.reserve( totalVertices );

	std::size_t currentIndex = 0;
	for( auto const& s : shapes )
	{
		auto const& objMesh = s.mesh;

		if( objMesh.indices.empty() )
			continue;

		assert( !objMesh.material_ids.empty() );

		// The OBJ format allows for general polygons and not just triangles.
		// However, this code only deals with triangles. Check that we only
		// have triangles (debug mode only).
		assert( objMesh.indices.size() % 3 == 0 );
#		ifndef NDEBUG
		for( auto faceVerts : objMesh.num_face_vertices )
			assert( 3 == faceVerts );
#		endif // ~ NDEBUG

		// Each of our rendered objMeshes can only have a single material. 
		// Split the OBJ objMesh into multiple objMeshes if there are multiple 
		// materials. 
		//
		// Note: if a single material is repeated multiple times, this will
		// generate a new objMesh for each time the material is encountered.
		int currentMaterial = objMesh.material_ids[0]-1; // start a new material!

		std::size_t indices = 0;

		std::size_t face = 0, vert = 0;
		for( auto const& objIdx : objMesh.indices )
		{
			// check if material changed
			auto const matId = objMesh.material_ids[face];
			if( matId != currentMaterial )
			{
				if( indices )
				{
					assert( currentMaterial >= 0 ); 

					MeshInfo mesh{};
					mesh.materialIndex     = currentMaterial;
					mesh.meshName          = s.name + "::" + model.materials[currentMaterial].materialName;
					mesh.vertexStartIndex  = currentIndex;
					mesh.numberOfVertices  = indices;

					model.meshes.emplace_back( mesh );
				}

				currentIndex += indices;
				currentMaterial = matId;
				indices = 0;
			}

			// copy over data
			model.vertexPositions.emplace_back( glm::vec3(
				attrib.vertices[ objIdx.vertex_index * 3 + 0 ],
				attrib.vertices[ objIdx.vertex_index * 3 + 1 ],
				attrib.vertices[ objIdx.vertex_index * 3 + 2 ]
			) );

			assert( objIdx.normal_index >= 0 ); // must have a normal!
			model.vertexNormals.emplace_back( glm::vec3(
				attrib.normals[ objIdx.normal_index * 3 + 0 ],
				attrib.normals[ objIdx.normal_index * 3 + 1 ],
				attrib.normals[ objIdx.normal_index * 3 + 2 ]
			) );

			if( objIdx.texcoord_index >= 0 )
			{
				model.vertexTextureCoords.emplace_back( glm::vec2(
					attrib.texcoords[ objIdx.texcoord_index * 2 + 0 ],
					attrib.texcoords[ objIdx.texcoord_index * 2 + 1 ]
				) );
			}
			else
			{
				model.vertexTextureCoords.emplace_back( glm::vec2( 0.f, 0.f ) );
			}

			++indices;

			// accounting: next vertex
			++vert;
			if( 3 == vert )
			{
				++face;
				vert = 0;
			}
		}

		if( indices )
		{
			assert( currentMaterial >= 0 ); 

			MeshInfo mesh{};
			mesh.materialIndex     = currentMaterial;
			mesh.meshName          = s.name + "::" + model.materials[currentMaterial].materialName;
			mesh.vertexStartIndex  = currentIndex;
			mesh.numberOfVertices  = indices;

			currentIndex += indices;

			model.meshes.emplace_back( mesh );
		}
	}

	assert( model.vertexPositions.size() == totalVertices );
	assert( model.vertexNormals.size() == totalVertices );
	assert( model.vertexTextureCoords.size() == totalVertices );
	
	return model;
}

LoadedMesh create_loaded_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator,
	lut::DescriptorPool& dpool, lut::DescriptorSetLayout& objectLayout, ModelData const& model)
{
	std::vector<labutils::Buffer> vertices;
	std::vector<labutils::Buffer> vertexNormals;
	std::vector<labutils::Buffer> textureCoords;
	std::vector<labutils::Buffer> vertexColor;

	std::vector<bool> useTexture;
	std::vector<std::string> materiaPath;

	std::vector<std::uint32_t> vertexCount;

	// Create default texture sampler for later use in texture
	lut::Sampler defaultSampler = lut::create_default_sampler(aContext);

	for (size_t i = 0; i < model.meshes.size(); i++)
	{
		std::size_t vertexStartIndex = model.meshes[i].vertexStartIndex;
		std::size_t numberOfVertices = model.meshes[i].numberOfVertices;

		std::vector<glm::vec3> positions(&model.vertexPositions[vertexStartIndex], 
			&model.vertexPositions[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec3> normals(&model.vertexNormals[vertexStartIndex],
			&model.vertexNormals[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec2> texCoords(&model.vertexTextureCoords[vertexStartIndex],
			&model.vertexTextureCoords[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec3> colour;
		for (size_t j = vertexStartIndex; j < vertexStartIndex + numberOfVertices; j++)
		{
			colour.push_back(model.materials[model.meshes[i].materialIndex].color);
		}

		if (model.materials[model.meshes[i].materialIndex].colorTexturePath != "")
		{
			materiaPath.push_back(model.materials[model.meshes[i].materialIndex].colorTexturePath);
			useTexture.push_back(true);
		}
		else
		{
			materiaPath.push_back("");
			useTexture.push_back(false);
		}

		vertexCount.push_back(numberOfVertices - 1);

		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * positions.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexNormalGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * normals.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexTexCoordsGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec2) * texCoords.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexColorsGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * colour.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * positions.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer normalStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * normals.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer textCoordStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec2) * texCoords.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer colorStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * colour.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(posPtr, positions.data(), sizeof(glm::vec3) * positions.size());
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		void* normalPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, normalStaging.allocation, &normalPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(normalPtr, normals.data(), sizeof(glm::vec3) * normals.size());
		vmaUnmapMemory(aAllocator.allocator, normalStaging.allocation);

		void* textCoordPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, textCoordStaging.allocation, &textCoordPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(textCoordPtr, texCoords.data(), sizeof(glm::vec2) * texCoords.size());
		vmaUnmapMemory(aAllocator.allocator, textCoordStaging.allocation);

		void* colorPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, colorStaging.allocation, &colorPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(colorPtr, colour.data(), sizeof(glm::vec3) * colour.size());
		vmaUnmapMemory(aAllocator.allocator, colorStaging.allocation);

		// Make sure the vulkan instance is still alive after all transfers are completed.
		lut::Fence uploadComplete = lut::create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers
		lut::CommandPool uploadPool = lut::create_command_pool(aContext);
		VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy positionCopy{};
		positionCopy.size = sizeof(glm::vec3) * positions.size();

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer,
			1, &positionCopy);

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy normalCopy{};
		normalCopy.size = sizeof(glm::vec3) * normals.size();

		vkCmdCopyBuffer(uploadCmd, normalStaging.buffer, vertexNormalGPU.buffer,
			1, &normalCopy);

		lut::buffer_barrier(uploadCmd,
			vertexNormalGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy texCoordCopy{};
		texCoordCopy.size = sizeof(glm::vec2) * texCoords.size();

		vkCmdCopyBuffer(uploadCmd, textCoordStaging.buffer, vertexTexCoordsGPU.buffer,
			1, &texCoordCopy);

		lut::buffer_barrier(uploadCmd,
			vertexTexCoordsGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy colorCopy{};
		colorCopy.size = sizeof(glm::vec3) * colour.size();

		vkCmdCopyBuffer(uploadCmd, colorStaging.buffer, vertexColorsGPU.buffer,
			1, &colorCopy);

		lut::buffer_barrier(uploadCmd,
			vertexColorsGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Submit transfer commands
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1,
			&submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n"
				"vkQueueSubmit returned() %s", lut::to_string(res).c_str());
		}

		// Wait for command to finish before we destroy the temporary resources
		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
			VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n"
				"vkWaitForFence() returned %s", lut::to_string(res).c_str());
		}

		vertices.push_back(std::move(vertexPosGPU));
		vertexNormals.push_back(std::move(vertexNormalGPU));
		textureCoords.push_back(std::move(vertexTexCoordsGPU));
		vertexColor.push_back(std::move(vertexColorsGPU));
	}

	return LoadedMesh
	{
		std::move(vertices),
		std::move(vertexNormals),
		std::move(textureCoords),
		std::move(vertexColor),

		std::move(useTexture),
		std::move(materiaPath),
		
		std::move(vertexCount)
	};
}

LoadedMesh load_to_vertex_buffer(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator,
	lut::DescriptorPool& dpool, lut::DescriptorSetLayout& objectLayout, ModelData& carModel,
	ModelData& cityModel)
{
	std::vector<labutils::Buffer> vertices;
	std::vector<labutils::Buffer> vertexNormals;
	std::vector<labutils::Buffer> textureCoords;
	std::vector<labutils::Buffer> vertexColor;

	std::vector<bool> useTexture;
	std::vector<std::string> materiaPath;

	std::vector<std::uint32_t> vertexCount;

	// Create default texture sampler for later use in texture
	lut::Sampler defaultSampler = lut::create_default_sampler(aContext);

	// Load car data first
	for (size_t i = 0; i < carModel.meshes.size(); i++)
	{
		std::size_t vertexStartIndex = carModel.meshes[i].vertexStartIndex;
		std::size_t numberOfVertices = carModel.meshes[i].numberOfVertices;

		std::vector<glm::vec3> positions(&carModel.vertexPositions[vertexStartIndex],
			&carModel.vertexPositions[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec3> normals(&carModel.vertexNormals[vertexStartIndex],
			&carModel.vertexNormals[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec2> texCoords(&carModel.vertexTextureCoords[vertexStartIndex],
			&carModel.vertexTextureCoords[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec3> colour;
		for (size_t j = vertexStartIndex; j < vertexStartIndex + numberOfVertices; j++)
		{
			colour.push_back(carModel.materials[carModel.meshes[i].materialIndex].color);
		}

		if (carModel.materials[carModel.meshes[i].materialIndex].colorTexturePath != "")
		{
			materiaPath.push_back(carModel.materials[carModel.meshes[i].materialIndex].colorTexturePath);
			useTexture.push_back(true);
		}
		else
		{
			materiaPath.push_back("");
			useTexture.push_back(false);
		}

		vertexCount.push_back(numberOfVertices - 1);

		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * positions.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexNormalGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * normals.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexTexCoordsGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec2) * texCoords.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexColorsGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * colour.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * positions.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer normalStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * normals.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer textCoordStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec2) * texCoords.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer colorStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * colour.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(posPtr, positions.data(), sizeof(glm::vec3) * positions.size());
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		void* normalPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, normalStaging.allocation, &normalPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(normalPtr, normals.data(), sizeof(glm::vec3) * normals.size());
		vmaUnmapMemory(aAllocator.allocator, normalStaging.allocation);

		void* textCoordPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, textCoordStaging.allocation, &textCoordPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(textCoordPtr, texCoords.data(), sizeof(glm::vec2) * texCoords.size());
		vmaUnmapMemory(aAllocator.allocator, textCoordStaging.allocation);

		void* colorPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, colorStaging.allocation, &colorPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(colorPtr, colour.data(), sizeof(glm::vec3) * colour.size());
		vmaUnmapMemory(aAllocator.allocator, colorStaging.allocation);

		// Make sure the vulkan instance is still alive after all transfers are completed.
		lut::Fence uploadComplete = lut::create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers
		lut::CommandPool uploadPool = lut::create_command_pool(aContext);
		VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy positionCopy{};
		positionCopy.size = sizeof(glm::vec3) * positions.size();

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer,
			1, &positionCopy);

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy normalCopy{};
		normalCopy.size = sizeof(glm::vec3) * normals.size();

		vkCmdCopyBuffer(uploadCmd, normalStaging.buffer, vertexNormalGPU.buffer,
			1, &normalCopy);

		lut::buffer_barrier(uploadCmd,
			vertexNormalGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy texCoordCopy{};
		texCoordCopy.size = sizeof(glm::vec2) * texCoords.size();

		vkCmdCopyBuffer(uploadCmd, textCoordStaging.buffer, vertexTexCoordsGPU.buffer,
			1, &texCoordCopy);

		lut::buffer_barrier(uploadCmd,
			vertexTexCoordsGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy colorCopy{};
		colorCopy.size = sizeof(glm::vec3) * colour.size();

		vkCmdCopyBuffer(uploadCmd, colorStaging.buffer, vertexColorsGPU.buffer,
			1, &colorCopy);

		lut::buffer_barrier(uploadCmd,
			vertexColorsGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Submit transfer commands
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1,
			&submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n"
				"vkQueueSubmit returned() %s", lut::to_string(res).c_str());
		}

		// Wait for command to finish before we destroy the temporary resources
		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
			VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n"
				"vkWaitForFence() returned %s", lut::to_string(res).c_str());
		}

		vertices.push_back(std::move(vertexPosGPU));
		vertexNormals.push_back(std::move(vertexNormalGPU));
		textureCoords.push_back(std::move(vertexTexCoordsGPU));
		vertexColor.push_back(std::move(vertexColorsGPU));
	}

	// Now load the city
	for (size_t i = 0; i < cityModel.meshes.size(); i++)
	{
		std::size_t vertexStartIndex = cityModel.meshes[i].vertexStartIndex;
		std::size_t numberOfVertices = cityModel.meshes[i].numberOfVertices;

		std::vector<glm::vec3> positions(&cityModel.vertexPositions[vertexStartIndex],
			&cityModel.vertexPositions[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec3> normals(&cityModel.vertexNormals[vertexStartIndex],
			&cityModel.vertexNormals[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec2> texCoords(&cityModel.vertexTextureCoords[vertexStartIndex],
			&cityModel.vertexTextureCoords[vertexStartIndex + numberOfVertices - 1]);

		std::vector<glm::vec3> colour;
		for (size_t j = vertexStartIndex; j < vertexStartIndex + numberOfVertices; j++)
		{
			colour.push_back(cityModel.materials[cityModel.meshes[i].materialIndex].color);
		}

		if (cityModel.materials[cityModel.meshes[i].materialIndex].colorTexturePath != "")
		{
			materiaPath.push_back(cityModel.materials[cityModel.meshes[i].materialIndex].colorTexturePath);
			useTexture.push_back(true);
		}
		else
		{
			materiaPath.push_back("");
			useTexture.push_back(false);
		}

		vertexCount.push_back(numberOfVertices - 1);

		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * positions.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexNormalGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * normals.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexTexCoordsGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec2) * texCoords.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer vertexColorsGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * colour.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * positions.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer normalStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * normals.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer textCoordStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec2) * texCoords.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer colorStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * colour.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(posPtr, positions.data(), sizeof(glm::vec3) * positions.size());
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		void* normalPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, normalStaging.allocation, &normalPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(normalPtr, normals.data(), sizeof(glm::vec3) * normals.size());
		vmaUnmapMemory(aAllocator.allocator, normalStaging.allocation);

		void* textCoordPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, textCoordStaging.allocation, &textCoordPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(textCoordPtr, texCoords.data(), sizeof(glm::vec2) * texCoords.size());
		vmaUnmapMemory(aAllocator.allocator, textCoordStaging.allocation);

		void* colorPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, colorStaging.allocation, &colorPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(colorPtr, colour.data(), sizeof(glm::vec3) * colour.size());
		vmaUnmapMemory(aAllocator.allocator, colorStaging.allocation);

		// Make sure the vulkan instance is still alive after all transfers are completed.
		lut::Fence uploadComplete = lut::create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers
		lut::CommandPool uploadPool = lut::create_command_pool(aContext);
		VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy positionCopy{};
		positionCopy.size = sizeof(glm::vec3) * positions.size();

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer,
			1, &positionCopy);

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy normalCopy{};
		normalCopy.size = sizeof(glm::vec3) * normals.size();

		vkCmdCopyBuffer(uploadCmd, normalStaging.buffer, vertexNormalGPU.buffer,
			1, &normalCopy);

		lut::buffer_barrier(uploadCmd,
			vertexNormalGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy texCoordCopy{};
		texCoordCopy.size = sizeof(glm::vec2) * texCoords.size();

		vkCmdCopyBuffer(uploadCmd, textCoordStaging.buffer, vertexTexCoordsGPU.buffer,
			1, &texCoordCopy);

		lut::buffer_barrier(uploadCmd,
			vertexTexCoordsGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		VkBufferCopy colorCopy{};
		colorCopy.size = sizeof(glm::vec3) * colour.size();

		vkCmdCopyBuffer(uploadCmd, colorStaging.buffer, vertexColorsGPU.buffer,
			1, &colorCopy);

		lut::buffer_barrier(uploadCmd,
			vertexColorsGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Submit transfer commands
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1,
			&submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n"
				"vkQueueSubmit returned() %s", lut::to_string(res).c_str());
		}

		// Wait for command to finish before we destroy the temporary resources
		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
			VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n"
				"vkWaitForFence() returned %s", lut::to_string(res).c_str());
		}

		vertices.push_back(std::move(vertexPosGPU));
		vertexNormals.push_back(std::move(vertexNormalGPU));
		textureCoords.push_back(std::move(vertexTexCoordsGPU));
		vertexColor.push_back(std::move(vertexColorsGPU));
	}

	return LoadedMesh
	{
		std::move(vertices),
		std::move(vertexNormals),
		std::move(textureCoords),
		std::move(vertexColor),

		std::move(useTexture),
		std::move(materiaPath),

		std::move(vertexCount)
	};
}
