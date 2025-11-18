/*
 * Copyright 2024 Edward Guy Capriolo
 *
 * The Deliverance Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package io.teknek.deliverance.model.qwen2;


import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class Qwen2ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Qwen2Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Qwen2Config.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return Qwen2Tokenizer.class;
    }
}
