###### BLIP
BLIP: 引导语言图像预训练（如上图所示），以实现统一的视觉语言理解和生成。BLIP是一个新的VLP框架，与现有方法相比，它可以实现更广泛的下游任务。它分别从模型和数据角度有两个贡献:

（1） 多模态编码器-解码器混合（MED）：一种用于有效多任务预训练和灵活迁移学习的新模型架构。MED可以作为单模态编码器、基于图像的文本编码器或基于图像的文本解码器工作。该模型与三个视觉语言目标联合预训练：图像文本对比学习、图像文本匹配和图像条件语言建模。

（2） 字幕和过滤（CapFilt）：一种新的数据集增强方法，用于从噪声图像-文本对中学习。作者将预先训练的MED分为两个模块: 一个字幕器，用于生成给定web图像的合成字幕，以及一个过滤器，用于从原始web文本和合成文本中删除嘈杂的字幕。

