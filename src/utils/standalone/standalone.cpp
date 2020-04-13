// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <stdlib.h>
#include <chrono>
#include <vector>
#include <fstream>

#include <Magnum/configure.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/Timeline.h>

#include "esp/assets/ResourceManager.h"
#include "esp/gfx/Renderer.h"
#include "esp/gfx/RenderCamera.h"
#include "esp/nav/PathFinder.h"
#include "esp/physics/PhysicsManager.h"
#include "esp/physics/RigidObject.h"
#include "esp/scene/ObjectControls.h"
#include "esp/scene/SceneManager.h"
#include "esp/scene/SceneNode.h"

#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/String.h>
#include <Magnum/DebugTools/Screenshot.h>
#include <Magnum/EigenIntegration/GeometryIntegration.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <sophus/so3.hpp>
#include "esp/core/esp.h"
#include "esp/gfx/Drawable.h"
#include "esp/io/io.h"

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/BufferImage.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>

#include "esp/scene/SceneConfiguration.h"
#include "esp/sim/Simulator.h"

#include "esp/gfx/configure.h"
#include <vector>
#include "renderdoc.h"
#include <dlfcn.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "stb_image_write.h"
#include <pthread.h>
#include <atomic>
#include <thread>

using namespace Magnum;
using namespace Math::Literals;
using namespace Corrade;

using namespace esp;
using namespace esp::gfx;

class RenderDoc {
public:
    RenderDoc();

    inline void startFrame() const
    {
        if (rdoc_impl_) startImpl();
    }

    inline void endFrame() const
    {
        if (rdoc_impl_) endImpl();
    }

private:
    void startImpl() const;
    void endImpl() const;

    void *rdoc_impl_;
};


using RenderDocApi = const RENDERDOC_API_1_4_1 *;

static void *initRDoc()
{
    void *lib = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (lib) {
        auto get_api = (pRENDERDOC_GetAPI)dlsym(lib, "RENDERDOC_GetAPI");

        void *ptrs;
        [[maybe_unused]] int ret =
            get_api(eRENDERDOC_API_Version_1_4_1, (void **)&ptrs);
        assert(ret == 1);

        return ptrs;
    } else {
        return nullptr;
    }
}

RenderDoc::RenderDoc()
    : rdoc_impl_(initRDoc())
{}

void RenderDoc::startImpl() const
{
    ((RenderDocApi)rdoc_impl_)->StartFrameCapture(nullptr, nullptr);
}

void RenderDoc::endImpl() const
{
    ((RenderDocApi)rdoc_impl_)->EndFrameCapture(nullptr, nullptr);
}

namespace Cr = Corrade;
namespace Mn = Magnum;

const Mn::GL::Framebuffer::ColorAttachment RgbaBuffer =
    Mn::GL::Framebuffer::ColorAttachment{0};
const Mn::GL::Framebuffer::ColorAttachment UnprojectedDepthBuffer =
    Mn::GL::Framebuffer::ColorAttachment{0};

struct NoObjectRenderTarget {
  NoObjectRenderTarget(const Mn::Vector2i& size,
       const Mn::Vector2& depthUnprojection,
       DepthShader* depthShader)
      : colorBuffer_{},
        depthRenderTexture_{},
        framebuffer_{Mn::NoCreate},
        depthUnprojection_{depthUnprojection},
        depthShader_{depthShader},
        unprojectedDepth_{Mn::NoCreate},
        depthUnprojectionMesh_{Mn::NoCreate},
        depthUnprojectionFrameBuffer_{Mn::NoCreate} {
    if (depthShader_) {
      CORRADE_INTERNAL_ASSERT(depthShader_->flags() &
                              DepthShader::Flag::UnprojectExistingDepth);
    }

    colorBuffer_.setStorage(Mn::GL::RenderbufferFormat::SRGB8Alpha8, size);
    depthRenderTexture_.setMinificationFilter(Mn::GL::SamplerFilter::Nearest)
        .setMagnificationFilter(Mn::GL::SamplerFilter::Nearest)
        .setWrapping(Mn::GL::SamplerWrapping::ClampToEdge)
        .setStorage(1, Mn::GL::TextureFormat::DepthComponent32F, size);

    framebuffer_ = Mn::GL::Framebuffer{{{}, size}};
    framebuffer_.attachRenderbuffer(RgbaBuffer, colorBuffer_)
        .attachTexture(Mn::GL::Framebuffer::BufferAttachment::Depth,
                       depthRenderTexture_, 0)
        .mapForDraw({{0, RgbaBuffer}});
    CORRADE_INTERNAL_ASSERT(
        framebuffer_.checkStatus(Mn::GL::FramebufferTarget::Draw) ==
        Mn::GL::Framebuffer::Status::Complete);
  }

  void initDepthUnprojector() {
    if (depthUnprojectionMesh_.id() == 0) {
      unprojectedDepth_ = Mn::GL::Renderbuffer{};
      unprojectedDepth_.setStorage(Mn::GL::RenderbufferFormat::R32F,
                                   framebufferSize());

      depthUnprojectionFrameBuffer_ =
          Mn::GL::Framebuffer{{{}, framebufferSize()}};
      depthUnprojectionFrameBuffer_
          .attachRenderbuffer(UnprojectedDepthBuffer, unprojectedDepth_)
          .mapForDraw({{0, UnprojectedDepthBuffer}});
      CORRADE_INTERNAL_ASSERT(
          framebuffer_.checkStatus(Mn::GL::FramebufferTarget::Draw) ==
          Mn::GL::Framebuffer::Status::Complete);

      depthUnprojectionMesh_ = Mn::GL::Mesh{};
      depthUnprojectionMesh_.setCount(3);
    }
  }

  void unprojectDepthGPU() {
    CORRADE_INTERNAL_ASSERT(depthShader_ != nullptr);
    initDepthUnprojector();

    depthUnprojectionFrameBuffer_.bind();
    depthShader_->bindDepthTexture(depthRenderTexture_)
        .setDepthUnprojection(depthUnprojection_);

    depthUnprojectionMesh_.draw(*depthShader_);
  }

  void renderEnter() {
    framebuffer_.clearDepth(1.0);
    framebuffer_.clearColor(0, Mn::Color4{0, 0, 0, 1});
    framebuffer_.bind();
  }

  void renderExit() {}

  void blitRgbaToDefault() {
    framebuffer_.mapForRead(RgbaBuffer);
    ASSERT(framebuffer_.viewport() == Mn::GL::defaultFramebuffer.viewport());

    Mn::GL::AbstractFramebuffer::blit(
        framebuffer_, Mn::GL::defaultFramebuffer, framebuffer_.viewport(),
        Mn::GL::defaultFramebuffer.viewport(), Mn::GL::FramebufferBlit::Color,
        Mn::GL::FramebufferBlitFilter::Nearest);
  }

  void readFrameRgba(const Mn::MutableImageView2D& view) {
    framebuffer_.mapForRead(RgbaBuffer).read(framebuffer_.viewport(), view);
  }

  void readFrameDepth(const Mn::MutableImageView2D& view) {
    if (depthShader_) {
      unprojectDepthGPU();
      depthUnprojectionFrameBuffer_.mapForRead(UnprojectedDepthBuffer)
          .read(framebuffer_.viewport(), view);
    } else {
      Mn::MutableImageView2D depthBufferView{
          Mn::GL::PixelFormat::DepthComponent, Mn::GL::PixelType::Float,
          view.size(), view.data()};
      framebuffer_.read(framebuffer_.viewport(), depthBufferView);
      unprojectDepth(depthUnprojection_,
                     Cr::Containers::arrayCast<Mn::Float>(view.data()));
    }
  }

  Mn::Vector2i framebufferSize() const {
    return framebuffer_.viewport().size();
  }

  void readFrameRgbaGPU(uint8_t* devPtr) {
    // TODO: Consider implementing the GPU read functions with EGLImage
    // See discussion here:
    // https://github.com/facebookresearch/habitat-sim/pull/114#discussion_r312718502

    if (colorBufferCugl_ == nullptr)
      checkCudaErrors(cudaGraphicsGLRegisterImage(
          &colorBufferCugl_, colorBuffer_.id(), GL_RENDERBUFFER,
          cudaGraphicsRegisterFlagsReadOnly));

    checkCudaErrors(cudaGraphicsMapResources(1, &colorBufferCugl_, 0));

    cudaArray* array = nullptr;
    checkCudaErrors(
        cudaGraphicsSubResourceGetMappedArray(&array, colorBufferCugl_, 0, 0));
    const int widthInBytes = framebufferSize().x() * 4 * sizeof(uint8_t);
    checkCudaErrors(cudaMemcpy2DFromArray(devPtr, widthInBytes, array, 0, 0,
                                          widthInBytes, framebufferSize().y(),
                                          cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &colorBufferCugl_, 0));
  }

  void readFrameDepthGPU(float* devPtr) {
    unprojectDepthGPU();

    if (depthBufferCugl_ == nullptr)
      checkCudaErrors(cudaGraphicsGLRegisterImage(
          &depthBufferCugl_, unprojectedDepth_.id(), GL_RENDERBUFFER,
          cudaGraphicsRegisterFlagsReadOnly));

    checkCudaErrors(cudaGraphicsMapResources(1, &depthBufferCugl_, 0));

    cudaArray* array = nullptr;
    checkCudaErrors(
        cudaGraphicsSubResourceGetMappedArray(&array, depthBufferCugl_, 0, 0));
    const int widthInBytes = framebufferSize().x() * 1 * sizeof(float);
    checkCudaErrors(cudaMemcpy2DFromArray(devPtr, widthInBytes, array, 0, 0,
                                          widthInBytes, framebufferSize().y(),
                                          cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &depthBufferCugl_, 0));
  }

  ~NoObjectRenderTarget() {
    if (colorBufferCugl_ != nullptr)
      checkCudaErrors(cudaGraphicsUnregisterResource(colorBufferCugl_));
    if (depthBufferCugl_ != nullptr)
      checkCudaErrors(cudaGraphicsUnregisterResource(depthBufferCugl_));
  }

 private:
  Mn::GL::Renderbuffer colorBuffer_;
  Mn::GL::Texture2D depthRenderTexture_;
  Mn::GL::Framebuffer framebuffer_;

  Mn::Vector2 depthUnprojection_;
  DepthShader* depthShader_;
  Mn::GL::Renderbuffer unprojectedDepth_;
  Mn::GL::Mesh depthUnprojectionMesh_;
  Mn::GL::Framebuffer depthUnprojectionFrameBuffer_;

  cudaGraphicsResource_t colorBufferCugl_ = nullptr;
  cudaGraphicsResource_t depthBufferCugl_ = nullptr;
};

static RenderDoc rdoc;

struct State {
    WindowlessContext::uptr context;

    std::unique_ptr<assets::ResourceManager> resourceManager;
    std::unique_ptr<scene::SceneManager> sceneManager;
    scene::SceneGraph &scene;
    scene::SceneNode &rootNode;
    scene::SceneNode &navSceneNode;

    RenderCamera &cam;
    std::unique_ptr<DepthShader> depthShader;

    std::unique_ptr<NoObjectRenderTarget> tgt;
    uint8_t *color_dev_ptr;
    float *depth_dev_ptr;

    void draw(const Matrix4 &view)
    {
        tgt->renderEnter();
        cam.node().setTransformation(view);

        for (auto &it : scene.getDrawableGroups()) {
            it.second.prepareForDraw(cam);
            cam.draw(it.second, false);
        }

        tgt->readFrameDepthGPU(depth_dev_ptr);
        tgt->readFrameRgbaGPU(color_dev_ptr);
    }
};

void save_frames(const State &state)
{
    using namespace std;
    vector<uint8_t> color_buffer(256 * 256 * 4);
    vector<float> depth_buffer(256 * 256);

    cudaMemcpy(color_buffer.data(), state.color_dev_ptr, color_buffer.size() * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(depth_buffer.data(), state.depth_dev_ptr, depth_buffer.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    vector<uint8_t> color_flipped(256 * 256 * 4);
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            color_flipped[i * 256 * 4 + j * 4] = color_buffer[(255 - i) * 256 * 4 + j * 4];
            color_flipped[i * 256 * 4 + j * 4 + 1] = color_buffer[(255 - i) * 256 * 4 + j * 4 + 1];
            color_flipped[i * 256 * 4 + j * 4 + 2] = color_buffer[(255 - i) * 256 * 4 + j * 4 + 2];
            color_flipped[i * 256 * 4 + j * 4 + 3] = color_buffer[(255 - i) * 256 * 4 + j * 4 + 3];
        }
    }

    vector<uint8_t> depth_sdr(256 * 256);
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            float v = depth_buffer[(255 - i) * 256 + j];
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            depth_sdr[i * 256 + j] = v * 255;
        }
    }

    stbi_write_bmp("/tmp/m_color.bmp", 256, 256, 4, color_flipped.data());
    stbi_write_bmp("/tmp/m_depth.bmp", 256, 256, 1, depth_sdr.data());
}



State makeState(const std::string &scenepath)
{
    gfx::WindowlessContext::uptr context = gfx::WindowlessContext::create_unique(0);
    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);

    auto resourceManager = std::make_unique<assets::ResourceManager>();
    auto sceneManager = std::make_unique<scene::SceneManager>();

    int sceneID = sceneManager->initSceneGraph();

    scene::SceneGraph &sceneGraph = sceneManager->getSceneGraph(sceneID);
    scene::SceneNode &rootNode = sceneGraph.getRootNode();
    scene::SceneNode &navSceneNode = rootNode.createChild();
    const assets::AssetInfo info = assets::AssetInfo::fromPath(scenepath);

    auto &drawables = sceneGraph.getDrawables();
    resourceManager->loadScene(info, &navSceneNode, &drawables);
    gfx::RenderCamera &renderCamera = sceneGraph.getDefaultRenderCamera();
    renderCamera.setProjectionMatrix(256,
                                     256,
                                     0.01f,             // znear
                                     1000.0f,           // zfar
                                     90.0f);            // hfov

    renderCamera.setAspectRatioPolicy(
        Magnum::SceneGraph::AspectRatioPolicy::Extend);

    const Magnum::Matrix4 projection = Matrix4::perspectiveProjection(
                90.0_degf, static_cast<float>(256) / 256.f, 0.01f, 1000.f);

    Vector2 depth_unproject = gfx::calculateDepthUnprojection(projection);
    
    auto depthShader = std::make_unique<gfx::DepthShader>(gfx::DepthShader::Flag::UnprojectExistingDepth);

    auto tgt = std::make_unique<NoObjectRenderTarget>(Vector2i(256, 256), depth_unproject, depthShader.get());

    uint8_t *color_dev_ptr;
    cudaMalloc(&color_dev_ptr, sizeof(uint8_t)*4*256*256);
    float *depth_dev_ptr;
    cudaMalloc(&depth_dev_ptr, sizeof(float)*1*256*256);

    return State {
        std::move(context),
        std::move(resourceManager),
        std::move(sceneManager),
        sceneGraph,
        rootNode,
        navSceneNode,
        renderCamera,
        std::move(depthShader),
        std::move(tgt),
        color_dev_ptr,
        depth_dev_ptr
    };
}

constexpr size_t max_load_frames = 10000;
constexpr size_t max_render_frames = 10000;
constexpr int num_threads = 4;
constexpr bool debug = false;

std::vector<Matrix4> readViews(const char *dump_path)
{
    using namespace std;
    ifstream dump_file(dump_path, ios::binary);

    vector<Matrix4> views;

    for (size_t i = 0; i < max_load_frames; i++) {
        float raw[16];
        dump_file.read((char *)raw, sizeof(float)*16);
        views.emplace_back(
                Matrix4(Vector4(raw[0], raw[1], raw[2], raw[3]),
                        Vector4(raw[4], raw[5], raw[6], raw[7]),
                        Vector4(raw[8], raw[9], raw[10], raw[11]),
                        Vector4(raw[12], raw[13], raw[14], raw[15])));
    }

    return views;
}

int main(int argc, char *argv[])
{
    using namespace std;
    if (argc < 3) {
        std::cout << "SCENE VIEWS" << std::endl;
        return -1;
    }

    vector<Matrix4> init_views = readViews(argv[2]);
    size_t num_frames = min(init_views.size(), max_render_frames);

    pthread_barrier_t start_barrier;
    pthread_barrier_init(&start_barrier, nullptr, num_threads + 1);
    pthread_barrier_t end_barrier;
    pthread_barrier_init(&end_barrier, nullptr, num_threads + 1);

    vector<thread> threads;
    threads.reserve(num_threads);

    atomic_bool go(false);

    for (int t_idx = 0; t_idx < num_threads; t_idx++) {
        threads.emplace_back(
            [num_frames, &go, &start_barrier, &end_barrier]
            (const char *scene_path, vector<Matrix4> views)
            {
                State state = makeState(scene_path);

                random_device rd;
                mt19937 g(rd());
                shuffle(views.begin(), views.end(), g);

                pthread_barrier_wait(&start_barrier);
                while (!go.load()) {}

                for (size_t i = 0; i < num_frames; i++) {
                    const Matrix4 &mat = views[i];
                    state.draw(mat);
                }

                pthread_barrier_wait(&end_barrier);
            }, 
            argv[1], init_views);
    }

    pthread_barrier_wait(&start_barrier);
    if (debug) {
        rdoc.startFrame();
    }

    auto start = chrono::steady_clock::now();
    go.store(true);

    pthread_barrier_wait(&end_barrier);
    auto end = chrono::steady_clock::now();

    if (debug) {
        rdoc.endFrame();
    }

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "FPS: " << ((double)num_frames * num_threads / (double)diff.count()) * 1000.0 << endl;

    for (thread &t : threads) {
        t.join();
    }

    return 0;
}
