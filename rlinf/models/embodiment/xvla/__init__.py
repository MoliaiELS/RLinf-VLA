# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_model(cfg, torch_dtype=None):
    """Load a pinned X-VLA policy for RLinf's embodied actor and rollout workers."""
    from .policy import XVLAPolicy

    return XVLAPolicy.from_checkpoint(cfg, torch_dtype)
