const path = require('path');

module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Add alias to replace problematic three-mesh-bvh module with our patched version
      if (!webpackConfig.resolve) {
        webpackConfig.resolve = {};
      }
      
      if (!webpackConfig.resolve.alias) {
        webpackConfig.resolve.alias = {};
      }
      
      webpackConfig.resolve.alias['three-mesh-bvh/src/utils/ExtensionUtilities.js'] = 
        path.resolve(__dirname, './src/patches/three-mesh-bvh-patch.js');
      
      return webpackConfig;
    }
  }
};