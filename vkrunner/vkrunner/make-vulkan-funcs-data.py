#!/usr/bin/env python

# Copyright 2023 Neil Roberts

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

# This script is used to generate vulkan_funcs_data.rs. It is not run
# automatically as part of the build process but if need be it can be
# used to update the file as follows:
#
# ./make-vulkan-funcs-data.py > vulkan_funcs_data.rs

from mako.template import Template


CORE_FUNCS = [
    "vkGetInstanceProcAddr",
    "vkCreateInstance",
    "vkEnumerateInstanceExtensionProperties",
]


INSTANCE_FUNCS = [
    "vkCreateDevice",
    "vkDestroyInstance",
    "vkEnumerateDeviceExtensionProperties",
    "vkEnumeratePhysicalDevices",
    "vkGetDeviceProcAddr",
    "vkGetPhysicalDeviceFeatures",
    "vkGetPhysicalDeviceFeatures2KHR",
    "vkGetPhysicalDeviceFormatProperties",
    "vkGetPhysicalDeviceMemoryProperties",
    "vkGetPhysicalDeviceProperties",
    "vkGetPhysicalDeviceProperties2",
    "vkGetPhysicalDeviceQueueFamilyProperties",
    "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR",
]


DEVICE_FUNCS = [
    "vkAllocateCommandBuffers",
    "vkAllocateDescriptorSets",
    "vkAllocateMemory",
    "vkBeginCommandBuffer",
    "vkBindBufferMemory",
    "vkBindImageMemory",
    "vkCmdBeginRenderPass",
    "vkCmdBindDescriptorSets",
    "vkCmdBindIndexBuffer",
    "vkCmdBindPipeline",
    "vkCmdBindVertexBuffers",
    "vkCmdClearAttachments",
    "vkCmdCopyBufferToImage",
    "vkCmdCopyImageToBuffer",
    "vkCmdDispatch",
    "vkCmdDraw",
    "vkCmdDrawIndexed",
    "vkCmdDrawIndexedIndirect",
    "vkCmdEndRenderPass",
    "vkCmdPipelineBarrier",
    "vkCmdPushConstants",
    "vkCmdSetScissor",
    "vkCmdSetViewport",
    "vkCreateBuffer",
    "vkCreateCommandPool",
    "vkCreateComputePipelines",
    "vkCreateDescriptorPool",
    "vkCreateDescriptorSetLayout",
    "vkCreateFence",
    "vkCreateFramebuffer",
    "vkCreateGraphicsPipelines",
    "vkCreateImage",
    "vkCreateImageView",
    "vkCreatePipelineCache",
    "vkCreatePipelineLayout",
    "vkCreateRenderPass",
    "vkCreateSampler",
    "vkCreateSemaphore",
    "vkCreateShaderModule",
    "vkDestroyBuffer",
    "vkDestroyCommandPool",
    "vkDestroyDescriptorPool",
    "vkDestroyDescriptorSetLayout",
    "vkDestroyDevice",
    "vkDestroyFence",
    "vkDestroyFramebuffer",
    "vkDestroyImage",
    "vkDestroyImageView",
    "vkDestroyPipeline",
    "vkDestroyPipelineCache",
    "vkDestroyPipelineLayout",
    "vkDestroyRenderPass",
    "vkDestroySampler",
    "vkDestroySemaphore",
    "vkDestroyShaderModule",
    "vkEndCommandBuffer",
    "vkFlushMappedMemoryRanges",
    "vkFreeCommandBuffers",
    "vkFreeDescriptorSets",
    "vkFreeMemory",
    "vkGetBufferMemoryRequirements",
    "vkGetDeviceQueue",
    "vkGetImageMemoryRequirements",
    "vkGetImageSubresourceLayout",
    "vkInvalidateMappedMemoryRanges",
    "vkMapMemory",
    "vkQueueSubmit",
    "vkQueueWaitIdle",
    "vkResetFences",
    "vkUnmapMemory",
    "vkUpdateDescriptorSets",
    "vkWaitForFences",
]


TEMPLATE = """\
// Automatically generated by make-vulkan-funcs-data.py

#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct Library {
    lib_vulkan: *const c_void,
    lib_vulkan_is_fake: bool,

% for func in core_funcs:
    pub ${func}: vk::PFN_${func},
% endfor
}

#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct Instance {
% for func in instance_funcs:
    pub ${func}: vk::PFN_${func},
% endfor
}

#[derive(Debug, Clone)]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct Device {
% for func in device_funcs:
    pub ${func}: vk::PFN_${func},
% endfor
}

impl Instance {
    pub unsafe fn new(
        get_instance_proc_cb: GetInstanceProcFunc,
        user_data: *const c_void,
    ) -> Instance {
        Instance {
% for func in instance_funcs:
            ${func}: std::mem::transmute(get_instance_proc_cb(
                "${func}\\0".as_ptr().cast(),
                user_data,
            )),
% endfor
        }
    }
}

#[allow(dead_code)]
impl Device {
    pub fn new(instance: &Instance, device: vk::VkDevice) -> Device {
        Device {
% for func in device_funcs:
            ${func}: unsafe {
                std::mem::transmute(instance.vkGetDeviceProcAddr.unwrap()(
                    device,
                    "${func}\\0".as_ptr().cast(),
                ))
            },
% endfor
        }
    }
}
"""


def main():
    template = Template(TEMPLATE)
    print(template.render(core_funcs = CORE_FUNCS,
                          instance_funcs = INSTANCE_FUNCS,
                          device_funcs = DEVICE_FUNCS),
          end="")


if __name__ == '__main__':
    main()
