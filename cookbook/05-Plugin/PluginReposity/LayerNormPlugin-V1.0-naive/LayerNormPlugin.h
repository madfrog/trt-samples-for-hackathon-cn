#include <NvInfer.h>
#include <string>
#include <vector>

#define EPSILON 6e-6

// +------- Debug wrapper ------------------------------------------------------
#ifdef DEBUG
    #define WHERE_AM_I()                                 \
        do                                               \
        {                                                \
            printf("[%s]: this=->%p\n", __func__, this); \
        } while (0);
#else
    #define WHERE_AM_I()
#endif // #ifdef DEBUG

// +------- Plguin -------------------------------------------------------------
namespace
{
static const char *PLUGIN_NAME {"LayerNorm"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
// +------- Plugin body --------------------------------------------------------
class LayerNormPlugin : public IPluginV2DynamicExt
{
private:
    std::string name_;
    std::string namespace_;

public:
    LayerNormPlugin(const std::string &name):
        name_(name)
    {
        WHERE_AM_I();
    }

    LayerNormPlugin(const std::string &name, const void *data, size_t length):
        name_(name)
    {
        WHERE_AM_I();
    }

    LayerNormPlugin() = delete;

    ~LayerNormPlugin()
    {
        WHERE_AM_I();
    }

    size_t getSerializationSize() const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }

    void serialize(void *buffer) const noexcept override
    {
        WHERE_AM_I();
    }

    IPluginV2DynamicExt *clone() const noexcept override
    {
        WHERE_AM_I();
        return new LayerNormPlugin(name_);
    }

    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override
    {
        WHERE_AM_I();
        return inputs[0];
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
        if (inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

        bool res = false;
        switch (pos)
        {
        case 0:
            res = (inOut[pos].type == DataType::kFLOAT);
            break;
        case 1:
            res = inOut[pos].type == inOut[0].type;
            break;
        default: // should NOT be here!
            res = false;
        }
        return res;
    }

    DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const noexcept override
    {
        WHERE_AM_I();
        return DataType::kFLOAT;
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
    }

    size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }

    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }
    const char *getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }
    const char *getPluginType() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }
    const char *getPluginVersion() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    void terminate() noexcept override
    {
        WHERE_AM_I();
        return;
    }

    void destroy() noexcept override
    {
        WHERE_AM_I();
    }

    int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
}; // class LayerNormPlugin

class LayerNormPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    LayerNormPluginCreator()
    {
        WHERE_AM_I();
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    ~LayerNormPluginCreator() {}

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
    {
        WHERE_AM_I();
        return new LayerNormPlugin(name);
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        WHERE_AM_I();
        return new LayerNormPlugin(name, serialData, serialLength);
    }

    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }

    const char *getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }

    const char *getPluginName() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection *getFieldNames() noexcept override
    {
        WHERE_AM_I();
        return &fc_;
    }
}; // class LayerNormPluginCreator

} // namespace nvinfer1
